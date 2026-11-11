# YCB fruit scan attribution

Source: Yale-CMU-Berkeley (YCB) Object and Model Set.
Authors: Berk Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa, Pieter Abbeel, Aaron M. Dollar.
Dataset: https://ycb-benchmarks.s3.amazonaws.com/index.html
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
License text: https://creativecommons.org/licenses/by/4.0/legalcode
Accessed: 2026-11-04.

Included Google 16k objects: 011_banana, 012_strawberry, 013_apple, 014_lemon,
015_peach, 016_pear, 017_orange, 018_plum.
These are benchmark objects, not the organizer's competition assets.

Modifications: mesh vertices centred at source axis-aligned bounding-box centre and uniformly
scaled to development dimensions; original UV coordinates and texture PNG retained;
OBJ material directives removed in favour of explicit MuJoCo material binding;
convex collision meshes generated from overlapping vertex slabs along the longest axis.
Banana has 6 collision pieces, pear 3, other objects 1. This is an approximate collision
representation, not a certified shape decomposition or measured contact model.
Mass/friction are simulation assumptions. Original mesh/texture files and downloaded
archive SHA256 records are retained under references/ycb/.
Detailed transforms, dimensions and derived hashes: references/ycb/preparation_manifest.json.

Recommended citation: B. Calli et al., "The YCB object and Model set: Towards common benchmarks
for manipulation research," ICAR 2015, pp. 510–517, doi:10.1109/ICAR.2015.7251504.

Additional runtime modifications: optional per-instance HSV hue, saturation and brightness
augmentation in MuJoCo texture memory. Original PNG files remain unchanged. These colors
are synthetic variations of the same scan, not independent scans of differently colored objects.
