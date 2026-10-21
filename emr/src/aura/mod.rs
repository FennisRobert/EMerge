//! AURA (Adaptive Unstructured Refinement Algorithm): local tetrahedral mesh refinement and quality improvement.
pub mod geom;
#[allow(clippy::unnecessary_map_or)]
pub mod improve;
pub mod py;
#[allow(clippy::unnecessary_map_or)]
pub mod refine;
