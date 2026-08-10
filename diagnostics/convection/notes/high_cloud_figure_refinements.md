# Refinements to high-cloud figures produced by `refined_high_cloud_classification_diagnostics.py'

## Figure 1

- I'm not convinced the instruction to review the literature values in the `high_cloud_figure_review.md` was followed. It is not clear what the values convey at the moment, given that other than the TRMM value they do not match our classification. That suggests the classification is not doing a good job, but actually I think the values are just misleading. Do better here.
- Move the individual subplot legends into a single shared figure legend. The middle plot does not need its own legend labels; the horizontal axis label can just be changed to "Difference from control (% pts)".
- Change subplot (c) title to "Precipitating cloud"

## Figure 2
- See above comment about the literature values being out of context. The thin cirrus value agrees; but the deep-cloud criterion looks very different. Is that good, bad, or meaningless?

## Figure 3
- At present the distributions use 20 hPa bins. That is not a good choice given that pressure logarithmically decreases with height. It would be better to compute the histograms using well-resolved uniform height bins, then convert to pressure for plotting. That would give a more highly resolved histogram, too. Ensure the axes labels are appropriate for the histogram with non-uniform pressure bins.
- What is meant by "Deep-cloud boundary"? I think we decided 150 hPa is the start of the TTL. Ensure the legend labels are consistent with the scientific definitions in this work. 
- As above, the literature values are out of context and I am not sure what they add at present. I think it is useful to compare with the literature but only if it is a faithful comparison.
- The plot does not need to go any higher than ~60 hPa (or wherever the histogram values go to zero).

## Figure 4
- Now that the figure size is more appropriate, there is too much text in the title. Review the figure and check for yourself.
- When I said label the subplots, I meant by including (a), (b), etc in the subplot titles as in other plots.
- I don't like the cloud-top pressure subplot colourbar: I want something that gives a visual sense of depth.

## Figure 6
- The 400 hPa label is broken.

## Figure 7
- Splitting the single figure into multiple has helped with the aspect ratio of the lat/lon plots, but it is now hard to compare. Keep the existing plots as they are, but align the zonal mean latitude axis with the lat/lon plot, and duplicate the zonal mean plots together in one shared figure for comparison.


