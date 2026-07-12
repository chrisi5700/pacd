# Measurement / regression harness (not part of the shipped library). Decomposes
# the composite fixtures and reports coverage / spill / IoU per mesh, so a change
# to the fitter can be judged on real multi-part shapes instead of by eye.
target_add_executable(score-composites score_composites.cpp)
target_link_libraries(score-composites PRIVATE pacd_solver pacd_render fmt::fmt)
target_compile_features(score-composites PRIVATE cxx_std_23)
