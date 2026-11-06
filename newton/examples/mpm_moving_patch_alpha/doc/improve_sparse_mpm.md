>>>
If you want to use the sparse grid with cuda graphs you will need to set max_active_cell_count and for best scalability probably tune max_leaf_node_count,  max_lower_node_count and  max_upper_node_count, the defaults are quite conservative.
>>>

Could you figure out optimal parameters like `max_leaf_node_count`, `max_lower_node_count`, and `max_uppwer_node_count` for sparse grid to scale better with terrain size. 
So, ideally we should find parameter that shows this trend: as terrain size increases (1mx1m -> 2mx2m -> 3m x 3m -> ... -> 10mx10m), compute and VRAM ideally show constant quantit because in sparse mode, MPM grid is always built around active collision particles, so it should scale with terrain size.