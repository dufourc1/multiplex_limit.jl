using GLMakie



function visualise_pairs(proba_matrices = [P, P_permuted, P_continent],
        network_matrices = [A, A_permuted, A_continent],
        orderings = ["Original", "Permutation", "Continent"],
        layer_names = list_names,
        node_labels = [node_names, node_names_permuted, node_labels_continent],
        white_lines = [[0], [0], white_lines_continent])

    fig = Figure(size = (800, 400))
    menu_layer = Menu(fig[2, 1], options = zip(layer_names, 1:length(layer_names)), direction = :up)
    menu_ordering = Menu(fig[2, 2], options = zip(orderings, 1:length(orderings)), direction = :up)

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
    #node_labels_ordered = @lift node_labels[$(sg.sliders[2].value)]


    probs = @lift proba_matrices[$(menu_ordering.selection)][:, :, $(menu_layer.selection)]
    network = @lift network_matrices[$(menu_ordering.selection)][
        :, :, $(menu_layer.selection)]
    lines = @lift white_lines[$(menu_ordering.selection)]






    ax_network, hm_network = heatmap(fig[1, 1],
        network, colorrange = (0, 1),
        colormap = :binary,
        #inspector_label = (self, i, p) -> "$(node_labels_ordered[][i[1]]) - $(node_labels_ordered[][i[2]]) ",
    )
    ax_probs, hm_probs = heatmap(fig[1, 2],
            probs, colorrange = (0, 1),
            colormap = :lipari,
            #inspector_label = (self, i, p) -> "$(node_labels_ordered[][i[1]]) - $(node_labels_ordered[][i[2]])",
    )
    #DataInspector(fig)
    vlines!(fig[1, 2], lines, color = :red)
    hlines!(fig[1, 2], lines, color = :red)


    hidedecorations!(ax_network)
    hidedecorations!(ax_probs)
    Colorbar(fig[1, 3], hm_probs, label = "Probability")

    colsize!(fig.layout, 3, Relative(0.1))
    colsize!(fig.layout, 1, Aspect(1,1))
    colsize!(fig.layout, 2, Aspect(1,1))

    return fig
end

visualise_pairs()
