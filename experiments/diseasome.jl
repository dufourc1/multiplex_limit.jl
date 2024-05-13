using NetworkHistogram, Statistics, CSV
using DataFrames
using ProgressMeter
using Makie, CairoMakie
using JLD
using MVBernoulli
using LinearAlgebra

Makie.inline!(true)
using Random
Random.seed!(123354472456192348634612304864326748)
include("utils.jl")
## Load the data


SAVE_FIG = false

path_to_data_folder = joinpath(@__DIR__, "../data/", "MultiplexDiseasome-master")

gwas_dataset = CSV.read(joinpath(path_to_data_folder, "Datasets", "DSG_network_from_GWAS.csv"), DataFrame)
omim_dataset = CSV.read(joinpath(path_to_data_folder, "Datasets", "DSG_network_from_OMIM.csv"), DataFrame)

##  dataset

function get_genotype_and_symptoms_cooccurrence_matrix(dataset::DataFrame, file::String)
    if isfile(file)
        loaded =  load(file)
        return loaded["node_names"], loaded["adj_genotype"], loaded["adj_symptoms"]
    end
    n = length(unique(dataset.disorder))
    diseases_names = sort(unique(dataset.disorder))

    adj_genotype = zeros(n, n)
    adj_symptoms = zeros(n, n)

    @showprogress for i in 1:n-1
        df_i = filter(x -> x.disorder == diseases_names[i], dataset)
        for j in i+1:n
            df_j = filter(x -> x.disorder == diseases_names[j], dataset)
            adj_genotype[i, j] = count_matches(df_i.gene_symb, df_j.gene_symb)
            adj_symptoms[i, j] = count_matches(df_i.symptom, df_j.symptom)
            adj_genotype[j, i] = adj_genotype[i, j]
            adj_symptoms[j, i] = adj_symptoms[i, j]
        end
    end

    save(file, "node_names" ,diseases_names, "adj_genotype" ,adj_genotype, "adj_symptoms" ,adj_symptoms)
    return diseases_names, adj_genotype, adj_symptoms
end

diseases_names, adj_genotype, adj_symptoms = get_genotype_and_symptoms_cooccurrence_matrix(omim_dataset, joinpath(path_to_data_folder,"adj_omim_2.jld"))




A_all = zeros(length(diseases_names), length(diseases_names), 2)
A_all[:, :, 1] =  adj_genotype
A_all[:, :, 2] = adj_symptoms



## Remove isolated diseases
A_all_binary = Int.(A_all .> 0)
degrees = dropdims(sum(A_all_binary, dims = (2)), dims = 2)

threshold = 2
#result with threshold = 2 and non_isolated_layer = findall(x -> x[1] ≥ threshold && x[2] ≥ threshold, eachrow(degrees))
non_isolated_layer = findall(x -> x[1] ≥ threshold && x[2] ≥ threshold, eachrow(degrees))

A_inter = A_all_binary[non_isolated_layer, non_isolated_layer, :]
A_weight_inter = A_all[non_isolated_layer, non_isolated_layer, :]
names = diseases_names[non_isolated_layer]
non_isolated_diseases = findall(x -> x ≥ threshold,
    vec(sum(A_inter, dims = (2, 3))))


A_weight = A_weight_inter[non_isolated_diseases, non_isolated_diseases, :]
names = names[non_isolated_diseases]
n = length(names)


#slow but can't be bothered to optimize
categories = Array{String}(undef, n)
for (i,name) in enumerate(names)
    category_raw = filter(x -> x.disorder == name, omim_dataset).disorder_cat
    if isempty(category_raw) || ismissing(category_raw[1])
        categories[i] = "z-Unknown"
    else
        categories[i] = category_raw[1]
    end
end

A = Int.(A_weight .> 0)



categories_degree = [mean(A[findall(categories .== cat), :, :]) for cat in unique(categories)]
categories_order = unique(categories)[sortperm(categories_degree, rev = true)]

degrees = vec(sum(A, dims = (2,3)))
tuple_cat_degree = [(findfirst(s -> s == categories[i], categories_order), degrees[i])
                    for i in 1:size(A, 1)]

sorting_by_category = sortperm(tuple_cat_degree,
    lt = (x, y) -> x[1] < y[1] ||
        (x[1] == y[1] && x[2] > y[2]))

## Convert to binary

fig = Figure()
ax = Axis(fig[1, 1], aspect = 1, title = "Genotype, threshold = $threshold")
ax2 = Axis(fig[1, 2], aspect = 1, title = "Symptoms, threshold = $threshold")
heatmap!(ax, A[:, :, 1], colormap = :binary)
heatmap!(ax2, A[:, :, 2], colormap = :binary)
display(fig)



## Fit the model
estimator, history = graphhist(A;
    starting_assignment_rule = EigenStart(),
    maxitr = Int(1e7),
    stop_rule = PreviousBestValue(10000))

fig = Figure()
best_ll = round(NetworkHistogram.get_bestitr(history)[2], sigdigits=2)
ax = Axis(fig[1, 1], xlabel = "Iterations", ylabel = "Log-likelihood", title = "Log-likelihood: $(best_ll)")
lines!(ax, get(history.history, :best_likelihood)...)
display(fig)


##
n_group_nodes = length(unique(estimator.node_labels))
max_shapes = n_group_nodes * (n_group_nodes + 1) ÷ 2
estimator_ = NetworkHistogram.GraphShapeHist(max_shapes, estimator)
for i in 1:n_group_nodes
    for j in i:n_group_nodes
        @assert all(estimator.θ[i,j,:] .== estimator_.θ[i,j])
    end
end


best_smoothed, bic_values = NetworkHistogram.get_best_smoothed_estimator(estimator, A)
display(lines(bic_values, legend = false, xlabel = "Number of shapes", ylabel = "BIC", title = "BIC values"))
##


estimated = best_smoothed
moments, indices = NetworkHistogram.get_moment_representation(estimated)

mvberns = MVBernoulli.from_tabulation.(estimated.θ)
marginals = MVBernoulli.marginals.(mvberns)
corrs = MVBernoulli.correlation_matrix.(mvberns)

mvberns_block = MVBernoulli.from_tabulation.(estimator_.θ)
marginals_block = MVBernoulli.marginals.(mvberns_block)
corrs_block = MVBernoulli.correlation_matrix.(mvberns_block)

@assert length(estimated.node_labels) == n
P_block = zeros(n, n, 3)
P_block[:, :, 1] = get_p_matrix([m[1] for m in marginals_block], estimator_.node_labels)
P_block[:, :, 2] = get_p_matrix([m[2] for m in marginals_block], estimator_.node_labels)
P_block[:, :, 3] = get_p_matrix([m[3] for m in corrs_block], estimator_.node_labels)

P = zeros(n, n, 3)
P[:, :, 1] = get_p_matrix([m[1] for m in marginals], estimated.node_labels)
P[:, :, 2] = get_p_matrix([m[2] for m in marginals], estimated.node_labels)
P[:, :, 3] = get_p_matrix([m[3] for m in corrs], estimated.node_labels)


#P[:,:,1:2] .= P[:,:,1:2] .^0.5
#P_block[:,:,1:2] .= P_block[:,:,1:2] .^0.5

function display_approx_and_data(P, A, sorting; label = "")
    fig = Figure(size = (700, 500))
    colormap = :lipari
    ax = Axis(fig[1, 1], aspect = 1, title = "Genotype layer", ylabel = "Histogram")
    ax2 = Axis(fig[1, 2], aspect = 1, title = "Phenotype layer")
    ax3 = Axis(fig[1, 3], aspect = 1, title = "Correlation")
    ylabel = label == "" ? "Adjacency matrix" : "Adjacency matrix (sorted by $label)"
    ax4 = Axis(fig[2, 1], aspect = 1, ylabel = ylabel)
    ax5 = Axis(fig[2, 2], aspect = 1)
    ax6 = Axis(fig[2, 3], aspect = 1)
    heatmap!(ax, P[sorting,sorting, 1], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax2, P[sorting,sorting, 2], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax3, P[sorting,sorting, 3], colormap = colormap, colorrange = (0, 1))
    heatmap!(ax4, A[sorting, sorting, 1], colormap = :binary)
    heatmap!(ax5, A[sorting, sorting, 2], colormap = :binary)
    heatmap!(
        ax6, A[sorting, sorting, 1] .* A[sorting, sorting, 2],
        colormap = :binary)
    Colorbar(fig[end+1, 1:3], colorrange = (0, 1),
        colormap = colormap, vertical = false, width = Relative(0.5), flipaxis = false)
    #Colorbar(fig[1:2, end + 1], colorrange = (-1, 1), label = "Correlation",
    #    colormap = :balance, vertical = true)
    hidedecorations!.([ax2, ax3, ax5, ax6])
    hidedecorations!.([ax, ax4], label=false)
    return fig
end


sorted_degree = sortperm(vec(sum(A, dims = (2, 3))), rev = true)
sorted_labels = sortperm(estimated.node_labels, rev = true, by = x ->(marginals[x,x][2],corrs[x,x][3],x))

sorted_groups = sortperm(1:length(unique(estimated.node_labels)), rev=true , by = x -> (marginals[x,x][2],x))

fig_fit = display_approx_and_data(P, A, sorted_labels, label = "")
rowgap!(fig_fit.layout, Relative(0.02))
colgap!(fig_fit.layout, Relative(0.01))
if SAVE_FIG
    save(joinpath(@__DIR__, "diseasome_fit.pdf"), fig_fit)
end
display(fig_fit)

##
A_plot_big = deepcopy(A)
A_plot_big[:, :, 1] .*= 1
A_plot_big[:, :, 2] .*= 2
A_plot = dropdims(sum(A_plot_big, dims = 3), dims = 3)

dict_name = Dict([0 => "None", 1 => "Genotype", 2 => "Phenotype", 3 => "Both"])
A_plot_string = [dict_name[a] for a in A_plot]

fig = Figure(size = (700, 400))
#titlelayout = GridLayout(fig[0, 2], halign = :center, tellwidth = false)
#Label(titlelayout[1,:], "Flattened multiplex adjacency matrix", halign = :center,
#    fontsize = 20)
#rowgap!(titlelayout, 0)

ax = Axis(fig[1, 1], aspect = 1, title = "Sorted by histogram clusters", titlesize = 14)
ax2 = Axis(fig[1, 2], aspect = 1, title = "Sorted by disease category", titlesize = 14)
ax3 = Axis(fig[1, 3], aspect = 1, title = "Sorted by degree", titlesize = 14)

heatmap!(ax, A_plot[sorted_labels, sorted_labels],
    colormap = Makie.Categorical(Reverse(:okabe_ito)))
heatmap!(ax2, A_plot[sorting_by_category, sorting_by_category],
    colormap = Makie.Categorical(Reverse(:okabe_ito)))
pl = heatmap!(ax3, A_plot[sorted_degree, sorted_degree],
    colormap = Makie.Categorical(Reverse(:okabe_ito)))
hidedecorations!.([ax, ax2, ax3], label = false)
#cb = Colorbar(fig[2, :],pl;
#    label = "Type of connection", vertical = false, width = Relative(0.5), flipaxis=false)
cb = Colorbar(fig[2, :];
    colormap = Reverse(cgrad(:okabe_ito, 4, categorical = true)),
    limits = (0,4),
    label = "Type of connection",
    labelsize = 14,
    ticklabelsize = 12,
    vertical = false, width = Relative(0.5), flipaxis=true, ticks = ([0.5, 1.5, 2.5, 3.5], ["None", "Genotype", "Phenotype", "Both"]))
display(fig)
rowgap!(fig.layout, 0.1)
fig
if SAVE_FIG
    save(joinpath(@__DIR__, "diseasome_adjacency.pdf"), fig)
end

##


display(display_approx_and_data(P, A, sorted_labels, label = "fit"))
display(display_approx_and_data(P, A, 1:n; label= "index"))
display(display_approx_and_data(P, A, sorting_by_category, label ="category"))
display(display_approx_and_data(P, A, sorted_degree, label = "degree"))
display(display_approx_and_data(P_block, A, 1:n; label = "index, block model"))
display(display_approx_and_data(P_block, A, sorting_by_category, label = "category, block model"))
display(display_approx_and_data(P_block, A, sorted_degree, label = "degree, block model"))
fig_block_model = display_approx_and_data(P_block, A, sorted_labels, label = "fit")
display(fig_block_model)
if SAVE_FIG
    save(joinpath(@__DIR__, "diseasome_fit_block_model.pdf"), fig_block_model)
end

## find interesting correlations


max_corr = maximum([c[3] for c in corrs if !isnan(c[3])])
corrs_values = sort([c[3] for c in corrs if !isnan(c[3])], rev = true)

indices_group = (findall(x -> x[3] == max_corr, corrs))
indices_node_group = [x[1] for x in indices_group]
indices_group = filter(x -> Tuple(x)[1] <= Tuple(x)[2], indices_group)


reverse_sort_group = sortperm(sorted_groups)
indices_group_in_picture = [(reverse_sort_group[Tuple(i)[1]],
                                reverse_sort_group[Tuple(i)[2]])
                            for i in indices_group]
fitted_dists = unique(mvberns[indices_group])

indices_node_group
nodes = [findall(estimated.node_labels .== i) for i in indices_node_group]
names_corr = [names[i] for i in nodes]
df_corr = filter(x -> x.disorder ∈ names_corr[1], omim_dataset)


indices_group = (findall(x -> x[3] >= corrs_values[2], corrs))
indices_node_group = [x[1] for x in indices_group]
indices_group = filter(x -> Tuple(x)[1] <= Tuple(x)[2], indices_group)

indices_group_in_picture = [(reverse_sort_group[Tuple(i)[1]],
                                reverse_sort_group[Tuple(i)[2]])
                            for i in indices_group]
fitted_dists = unique(mvberns[indices_group])

indices_node_group
nodes = [findall(x -> x == i || x == j,estimated.node_labels) for (i,j) in Tuple.(indices_group)]
names_corr = [names[i] for i in nodes]
df_corr = filter(x -> x.disorder ∈ names_corr[1], omim_dataset)

## find clique second layer

indices_group = (findall(x -> x[2] == 1, marginals))
indices_node_group = [x[1] for x in indices_group]
indices_group = filter(x -> Tuple(x)[1] <= Tuple(x)[2], indices_group)

reverse_sort_group = sortperm(sorted_groups)
indices_group_in_picture = [(reverse_sort_group[Tuple(i)[1]],
                                reverse_sort_group[Tuple(i)[2]])
                            for i in indices_group]
fitted_dists = unique(mvberns[indices_group])

indices_node_group
nodes = [findall(estimated.node_labels .== i) for i in indices_node_group]
names_marginal = [names[i] for i in nodes]
df_all_ones = filter(x -> x.disorder ∈ names_marginal[1], omim_dataset)
filter(x -> x.symptom == "cognitive impairment", df_all_ones)


##


communities_original_paper = [
    ["cutis laxa","syndromic intellectual disability","combined oxidative phosphorylation deficiency","congenital muscular dystrophy","
maple syrup urine disease","craniosynostosis","hereditary spastic paraplegia","congenital disorder of glycosylation type II","
lissencephaly","congenital disorder of glycosylation type I","Albright s hereditary osteodystrophy","Bamforth-Lazarus
syndrome","Borjeson-Forssman-Lehmann syndrome","Bowen-Conradi syndrome","Canavan disease","Charcot-Marie-Tooth
disease type X","Danon disease","Ellis-Van Creveld syndrome","Farber lipogranulomatosis","Greig cephalopolysyndactyly
syndrome","Hartnup disease","Joubert syndrome","L-2-hydroxyglutaric aciduria","Laron syndrome","Larsen syndrome","MASA
syndrome","Marshall-Smith syndrome","Perrault syndrome","Saethre-Chotzen syndrome","Sandhoff disease","Seckel syndrome","
X-linked ichthyosis","X-linked sideroblastic anemia with ataxia","acrofrontofacionasal dysostosis","acromesomelic dysplasia","
Hunter-Thompson type","alpha thalassemia","alpha-mannosidosis","argininosuccinic aciduria","beta-mannosidosis","
centronuclear myopathy","cerebrotendinous xanthomatosis","congenital disorder of glycosylation","creatine transporter
deficiency","dyskeratosis congenita","erythrokeratodermia variabilis","fibrodysplasia ossificans progressiva","galactosemia","
glycogen storage disease II","glycogen storage disease III","hemolytic-uremic syndrome","hydrocephalus","hyperargininemia","
hyperlysinemia","isovaleric acidemia","methylmalonic aciduria and homocystinuria type cblD","mevalonic aciduria","
microcephaly","microphthalmia","monilethrix","mucolipidosis","mucopolysaccharidosis II","mucosulfatidosis","multiple system
atrophy","myotonic dystrophy type 1","non-syndromic X-linked intellectual disability","olivopontocerebellar atrophy","orotic
aciduria","peroxisomal acyl-CoA oxidase deficiency","piebaldism","popliteal pterygium syndrome","propionic acidemia","
pseudohypoparathyroidism","pseudopseudohypoparathyroidism","sclerosteosis","spinocerebellar ataxia","syndromic X-linked
intellectual disability","thalassemia","tyrosinemia type II","Friedreich ataxia","MHC class II deficiency","Matthew-Wood syndrome","
premature ovarian failure","hypospadias","amyotrophic lateral sclerosis type 13","attention deficit hyperactivity disorder","
triosephosphate isomerase deficiency","congenital ichthyosis","congenital nystagmus","frontotemporal dementia","
pontocerebellar hypoplasia type 6","spinocerebellar ataxia type 5",],
    ["ovarian cancer","dyskeratosis congenita","piebaldism","LADD syndrome","Peutz-Jeghers syndrome","Noonan syndrome","Costello
syndrome","seborrheic keratosis","DNA ligase IV deficiency","LEOPARD syndrome","Li-Fraumeni syndrome","Lynch syndrome","
colorectal cancer","Rubinstein-Taybi syndrome","cardiofaciocutaneous syndrome","achondroplasia","autoimmune
lymphoproliferative syndrome","breast cancer","colon carcinoma","familial adenomatous polyposis","hemophagocytic
lymphohistiocytosis","leprosy","lung small cell carcinoma","mastocytosis","urticaria pigmentosa","pilomatrixoma","pancreatic
carcinoma","retinoblastoma","trilateral retinoblastoma","seminoma","testicular cancer","testicular germ cell cancer","stomach
cancer","acute myeloid leukemia","Birt-Hogg-Dube syndrome","lung cancer","epidermal nevus","urinary bladder cancer","
gastrointestinal stromal tumor","multiple myeloma","cervix carcinoma","thanatophoric dysplasia","adrenocortical carcinoma","
choroid plexus papilloma","hepatocellular carcinoma","malignant glioma","osteosarcoma","Proteus syndrome","ataxia
telangiectasia","non-Hodgkin lymphoma","embryonal rhabdomyosarcoma","von Hippel-Lindau disease","papillary renal cell
carcinoma","juvenile polyposis syndrome"],
    ["Andersen-Tawil syndrome","Robinow syndrome","long QT syndrome","Becker muscular dystrophy","Cockayne syndrome","
Duchenne muscular dystrophy","Emery-Dreifuss muscular dystrophy","Liddle syndrome","carnitine palmitoyltransferase II
deficiency","hemochromatosis","hypertrophic cardiomyopathy","scapuloperoneal myopathy","supravalvular aortic stenosis","
Brugada syndrome","atrial heart septal defect","familial atrial fibrillation","short QT syndrome","distal muscular dystrophy","
dilated cardiomyopathy","familial partial lipodystrophy","Jervell-Lange Nielsen syndrome","posterior polar cataract","
hyperaldosteronism","sick sinus syndrome","sudden infant death syndrome","progeria","arrhythmogenic right ventricular
cardiomyopathy","Wolff-Parkinson-White syndrome","restrictive cardiomyopathy","rippling muscle disease"],
    ["Walker-Warburg syndrome","nemaline myopathy","Aicardi-Goutieres syndrome","Gauchers disease","Meier-Gorlin syndrome","
coenzyme Q10 deficiency disease","congenital generalized lipodystrophy","congenital myasthenic syndrome","muscular
dystrophy-dystroglycanopathy","osteopetrosis","multiple system atrophy","Axenfeld-Rieger syndrome","craniometaphyseal
dysplasia","Barth syndrome","limb-girdle muscular dystrophy","paroxysmal nocturnal hemoglobinuria","Ullrich congenital
muscular dystrophy","oculopharyngeal muscular dystrophy","Fukuyama congenital muscular dystrophy","cytochrome-c
oxidase deficiency disease","cystinuria","visceral heterotaxy","von Willebrands disease","inclusion body myositis","glycogen
storage disease XV","pontocerebellar hypoplasia type 2A","pontocerebellar hypoplasia type 4","neutral lipid storage disease","
situs inversus"],
    ["Bothnia retinal dystrophy","cone-rod dystrophy","retinitis pigmentosa","Hirschsprungs disease","Waardenburgs syndrome","
Jervell-Lange Nielsen syndrome","Usher syndrome","Leber congenital amaurosis","congenital stationary night blindness","
persistent hyperplastic primary vitreous","striatonigral degeneration","achromatopsia","blue cone monochromacy","Stargardt
disease","bestrophinopathy","fundus albipunctatus","myopia","cone dystrophy","bradyopsia","cataract","posterior polar cataract","
hereditary night blindness","gyrate atrophy","vitelliform macular dystrophy","Aland Island eye disease","partial central choroid
dystrophy"],
    ["amyotrophic lateral sclerosis type 4","amyotrophic lateral sclerosis","amyotrophic neuralgia","essential tremor","Crouzon
syndrome","amyotrophic lateral sclerosis type 1","inclusion body myopathy with Paget disease of bone and frontotemporal
dementia","Gerstmann-Straussler-Scheinker syndrome","Niemann-Pick disease","Picks disease","amyotrophic lateral sclerosis
type 10","gangliosidosis GM1","amyotrophic lateral sclerosis type 11","amyotrophic lateral sclerosis type 12","amyotrophic
lateral sclerosis type 14","amyotrophic lateral sclerosis type 15","amyotrophic lateral sclerosis type 16","amyotrophic lateral
sclerosis type 17","amyotrophic lateral sclerosis type 18","amyotrophic lateral sclerosis type 19","amyotrophic lateral sclerosis
type 6","amyotrophic lateral sclerosis type 8","amyotrophic lateral sclerosis type 9","amyotrophic lateral sclerosis type 2","
primary open angle glaucoma"],
    ["autosomal recessive non-syndromic intellectual disability","Warburg micro syndrome","autosomal dominant non-syndromic
intellectual disability","infantile cerebellar-retinal degeneration","Andersen-Tawil syndrome","Cornelia de Lange syndrome","
Rothmund-Thomson syndrome","Bartter disease","Noonan syndrome","Opitz-GBBB syndrome","acromesomelic dysplasia","
Maroteaux type","sialuria","Fanconi syndrome","Lesch-Nyhan syndrome","Weill-Marchesani syndrome","asphyxiating thoracic
dystrophy","glycogen storage disease IX","intrahepatic cholestasis","mitochondrial complex V (ATP synthase) deficiency","
nuclear type 1","pontocerebellar hypoplasia type 2E","inclusion body myopathy with Paget disease of bone and
frontotemporal dementia","juvenile myelomonocytic leukemia","Sotos syndrome"],
    ["ACTH-secreting pituitary adenoma","Blau syndrome","Chediak-Higashi syndrome","Clouston syndrome","Hailey-Hailey disease","
Jobs syndrome","MHC class I deficiency","Papillon-Lefevre disease","Werner syndrome","acrodermatitis enteropathica","beta
thalassemia","biotinidase deficiency","cyclic hematopoiesis","hereditary sensory neuropathy","nevoid basal cell carcinoma
syndrome","oculocerebrorenal syndrome","port-wine stain","reticular dysgenesis","Hajdu-Cheney syndrome"],
    ["3MC syndrome","tetralogy of Fallot","oculodentodigital dysplasia","Alagille syndrome","sclerosteosis","syndactyly","
craniometaphyseal dysplasia","atrial heart septal defect","DiGeorge syndrome","Hajdu-Cheney syndrome","congenital
hypothyroidism","visceral heterotaxy","ventricular septal defect","double outlet right ventricle","velocardiofacial syndrome","
hypoplastic left heart syndrome","atrioventricular septal defect","chondrocalcinosis","congenital diaphragmatic hernia"],
    ["Bannayan-Riley-Ruvalcaba syndrome","Cowden disease","Peutz-Jeghers syndrome","xeroderma pigmentosum","endometrial
carcinoma","VACTERL association","basal ganglia calcification","hypospadias","familial meningioma","dysplastic nevus syndrome","
melanoma","skin melanoma","follicular thyroid carcinoma","head and neck squamous cell carcinoma","prostate cancer","
Kennedys disease","androgen insensitivity syndrome","photosensitive trichothiodystrophy"],
    ["centronuclear myopathy","Brown-Vialetto-Van Laere syndrome","Charcot-Marie-Tooth disease type 4","pontocerebellar
hypoplasia type 1B","Charcot-Marie-Tooth disease type 2","Charcot-Marie-Tooth disease intermediate type","Charcot-Marie-
Tooth disease type 1","distal hereditary motor neuropathy","distal muscular dystrophy","Gamstorp-Wohlfart syndrome","Krabbe
disease","Tangier disease","metachromatic leukodystrophy","Charcot-Marie-Tooth disease type 3","focal segmental
glomerulosclerosis","brachyolmia","Guillain-Barre syndrome"],
    ["3-M syndrome","achondrogenesis type II","spondylocostal dysostosis","Leigh disease","Walker-Warburg syndrome","
acrodysostosis","atelosteogenesis","holoprosencephaly","hypopituitarism","osteogenesis imperfecta","
otospondylomegaepiphyseal dysplasia","pre-eclampsia","Bjornstad syndrome","Bruck syndrome","multiple sclerosis","
cytochrome-c oxidase deficiency disease","schneckenbecken dysplasia"],
    ["Meckel syndrome","Joubert syndrome","renal-hepatic-pancreatic dysplasia","Bardet-Biedl syndrome","asphyxiating thoracic
dystrophy","autistic disorder","erythropoietic protoporphyria","generalized epilepsy with febrile seizures plus","nephrotic
syndrome","thrombophilia","triple-A syndrome","nephronophthisis","Senior-Loken syndrome","glycogen storage disease IV","
hypermethioninemia","orofaciodigital syndrome"],
    ["Aarskog-Scott syndrome","FG syndrome","Kallmann syndrome","Meckel syndrome","Ohdo syndrome","Prader-Willi syndrome","
Roberts syndrome","SC phocomelia syndrome","Simpson-Golabi-Behmel syndrome","Troyer syndrome","cold-induced sweating
syndrome","multiple synostoses syndrome","oculodentodigital dysplasia","tarsal-carpal coalition syndrome","synpolydactyly","
periventricular nodular heterotopia"],
    ["nemaline myopathy","distal arthrogryposis","Antley-Bixler syndrome","Carpenter syndrome","Loeys-Dietz syndrome","
brachydactyly","Bethlem myopathy","syndactyly","CHARGE syndrome","hydrolethalus syndrome","Down syndrome","multiple
intestinal atresia","persistent fetal circulation syndrome","Marfan syndrome","Smith-McCort dysplasia","proximal
symphalangism"],
]



for (index_corr,names) in enumerate(names_corr)
    for (index_com,community) in enumerate(communities_original_paper)
        count = count_matches(names, community)
        if count> 0.3*length(names)
            println(index_corr, " ", index_com)
            println(count," ", count/length(community), "  ", count/length(names))
            println(names)
            println(community)
        end
    end
end




for (index_corr,names) in enumerate(names_marginal)
    for (index_com,community) in enumerate(communities_original_paper)
        count = count_matches(names, community)
        if count> 0.1*length(names)
            println(index_corr, " ", index_com)
            println(count," ", count/length(community), "  ", count/length(names))
            println(names)
            println(community)
        end
    end
end
