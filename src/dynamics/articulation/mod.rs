//! Articulations using the reduced-coordinates formalism or using constraints.

pub use self::articulation::Articulation;
pub use self::articulation_motor::ArticulationMotor;
pub use self::cartesian_articulation::CartesianArticulation;
pub use self::fixed_articulation::FixedArticulation;
pub use self::free_articulation::FreeArticulation;
pub use self::prismatic_articulation::PrismaticArticulation;
pub use self::revolute_articulation::RevoluteArticulation;
pub use self::unit_articulation::UnitArticulation;
// pub use self::unit_articulation::{
//     unit_articulation_num_velocity_constraints, unit_articulation_position_constraint,
//     unit_articulation_velocity_constraints,
// };

#[cfg(feature = "dim3")]
pub use self::ball_articulation::BallArticulation;
#[cfg(feature = "dim3")]
pub use self::cylindrical_articulation::CylindricalArticulation;
#[cfg(feature = "dim3")]
pub use self::helical_articulation::HelicalArticulation;
#[cfg(feature = "dim3")]
pub use self::pin_slot_articulation::PinSlotArticulation;
#[cfg(feature = "dim3")]
pub use self::planar_articulation::PlanarArticulation;
#[cfg(feature = "dim3")]
pub use self::rectangular_articulation::RectangularArticulation;
#[cfg(feature = "dim3")]
pub use self::universal_articulation::UniversalArticulation;

mod articulation;
mod articulation_motor;
mod cartesian_articulation;
mod fixed_articulation;
mod free_articulation;
mod prismatic_articulation;
mod revolute_articulation;
mod unit_articulation;

#[cfg(feature = "dim3")]
mod ball_articulation;
#[cfg(feature = "dim3")]
mod cylindrical_articulation;
#[cfg(feature = "dim3")]
mod helical_articulation;
#[cfg(feature = "dim3")]
mod pin_slot_articulation;
#[cfg(feature = "dim3")]
mod planar_articulation;
#[cfg(feature = "dim3")]
mod rectangular_articulation;
#[cfg(feature = "dim3")]
mod universal_articulation;
