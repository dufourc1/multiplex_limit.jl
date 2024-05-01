using GLMakie



function visualise_pairs(proba_matrices ,
        network_matrices ,
        orderings,
        layer_names ,
        node_labels ,
        white_lines,
        continents_names)

    fig = Figure(size = (800, 400))
    menu_layer = Menu(fig[2, 1], options = zip(layer_names, 1:length(layer_names)), direction = :up)
    menu_ordering = Menu(fig[2, 2], options = zip(orderings, 1:length(orderings)), direction = :up, default = length(orderings))

    #sg = SliderGrid(fig[2, 1:3],
    #    (label = "Layer",range = 1:length(layer_names), startvalue = 1, format = x -> layer_names[x]),
    #    (label = "Sorting", range = 1:3, format = x -> orderings[x], startvalue = 2),
    #)

    #on(menu_layer.selection) do layer
    #    sg.sliders[1].value = layer
    #end
    #on(menu_ordering.selection) do ordering
    #    sg.sliders[2].value = ordering
    #end
    #probs = @lift proba_matrices[$(sg.sliders[2].value)][:, :, $(sg.sliders[1].value)]
    #network = @lift network_matrices[$(sg.sliders[2].value)][:, :, $(sg.sliders[1].value)]
    #lines = @lift white_lines[$(sg.sliders[2].value)]


    probs = @lift proba_matrices[$(menu_ordering.selection)][:, :, $(menu_layer.selection)]
    network = @lift network_matrices[$(menu_ordering.selection)][
        :, :, $(menu_layer.selection)]
    lines = @lift white_lines[$(menu_ordering.selection)]
    node_labels_ordered = @lift node_labels[$(menu_ordering.selection)]


    n = size(network_matrices[1], 1)
    xticks_pos_cont = Array{Float64}(undef, length(continents_names))
    xticks_pos_cont[1] = (1 + white_lines[3][1]) / 2
    xticks_pos_cont[end] = (white_lines[3][end]-0.5 + n+0.5) / 2
    for i in 2:length(continents_names)-1
        xticks_pos_cont[i] = (white_lines[3][i] + white_lines[3][i - 1]) / 2
    end
    println(xticks_pos_cont)

    dummy_ticks = [([],[]), ([],[]), (xticks_pos_cont, continents_names)]
    xticks =  @lift dummy_ticks[$(menu_ordering.selection)]







    ax_network, hm_network = heatmap(fig[1, 1],
        network, colorrange = (0, 1),
        colormap = :binary,
        inspector_label = (self, i, p) -> "$(node_labels_ordered[][i[1]]) - $(node_labels_ordered[][i[2]]) ",
    )
    ax_probs, hm_probs = heatmap(fig[1, 2],
            probs, colorrange = (0, 1),
            colormap = :lipari,
            axis = (xticks = xticks, yticks = xticks, xticklabelsize = 8, yticklabelsize = 8),
            inspector_label = (self, i, p) -> "$(node_labels_ordered[][i[1]]) - $(node_labels_ordered[][i[2]])",
    )
    DataInspector(fig)
    #vlines!(fig[1, 2], lines, color = :white)
    #hlines!(fig[1, 2], lines, color = :white)


    hidedecorations!(ax_network)
    hidedecorations!(ax_probs, ticklabels = false)
    Colorbar(fig[1, 3], hm_probs, label = "Probability")

    colsize!(fig.layout, 3, Relative(0.1))
    colsize!(fig.layout, 1, Aspect(1,1))
    colsize!(fig.layout, 2, Aspect(1,1))

    return fig
end
