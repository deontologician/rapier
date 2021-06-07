#![allow(missing_docs)] // For downcast.

use crate::dynamics::{Articulation, ArticulationMotor};
use crate::math::Real;
use downcast_rs::impl_downcast;

/// Trait implemented by articulations using the reduced-coordinates approach and allowing only one degree of freedom.
pub trait UnitArticulation: Articulation {
    /// The generalized coordinate of the unit articulation.
    fn position(&self) -> Real;
    /// The motor applied to the degree of freedom of the unit joint.
    fn motor(&self) -> &ArticulationMotor<Real>;
    /// The lower limit, if any, set to the generalized coordinate of this unit articulation.
    fn min_position(&self) -> Option<Real>;
    /// The upper limit, if any, set to the generalized coordinate of this unit articulation.
    fn max_position(&self) -> Option<Real>;
}

impl_downcast!(UnitArticulation);

/*
/// Computes the maximum number of velocity constraints to be applied by the given unit articulation.
pub fn unit_articulation_num_velocity_constraints<Real: RealField, J: UnitArticulation>(
    articulation: &J,
) -> usize {
    // FIXME: don't always keep the constraints active.
    let mut nconstraints = 0;

    if articulation.motor().enabled {
        nconstraints += 1;
    }
    if articulation.min_position().is_some() {
        nconstraints += 1;
    }
    if articulation.max_position().is_some() {
        nconstraints += 1;
    }

    nconstraints
}

/// Initializes and generate the velocity constraints applicable to the multibody links attached
/// to this articulation.
pub fn unit_articulation_velocity_constraints<Real: RealField, J: UnitArticulation>(
    articulation: &J,
    parameters: &IntegrationParameters,
    multibody: &Multibody<Real>,
    link: &MultibodyLink<Real>,
    assembly_id: usize,
    dof_id: usize,
    ext_vels: &[Real],
    ground_j_id: &mut usize,
    jacobians: &mut [Real],
    constraints: &mut ConstraintSet<Real, (), (), usize>,
) {
    let ndofs = multibody.ndofs();
    let impulses = multibody.impulses();
    let mut is_min_constraint_active = false;
    let articulation_velocity = multibody.articulation_velocity(link);

    let motor = articulation.motor();
    if motor.enabled {
        let dvel = articulation_velocity[dof_id] + ext_vels[link.assembly_id];

        DVectorSliceMut::from_slice(&mut jacobians[*ground_j_id..], ndofs).fill(Real::zero());
        jacobians[*ground_j_id + link.assembly_id + dof_id] = Real::one();

        let wj_id = *ground_j_id + ndofs;
        multibody.inv_mass_mul_unit_articulation_force(
            link,
            dof_id,
            Real::one(),
            &mut jacobians[wj_id..],
        );

        let inv_r = jacobians[wj_id + link.assembly_id + dof_id]; // = J^t * M^-1 J
        let velocity = motor
            .desired_velocity
            .clamp(-motor.max_velocity, motor.max_velocity);
        let rhs = dvel - velocity;
        let limits = motor.impulse_limits();
        let impulse_id = link.impulse_id + dof_id * 3;

        let constraint = BilateralGroundConstraint {
            impulse: impulses[impulse_id] * parameters.warmstart_coeff,
            r: Real::one() / inv_r,
            rhs,
            limits,
            impulse_id,
            assembly_id,
            j_id: *ground_j_id,
            wj_id: *ground_j_id + ndofs,
            ndofs,
        };

        constraints.velocity.bilateral_ground.push(constraint);
        *ground_j_id += 2 * ndofs;
    }

    if let Some(min_position) = articulation.min_position() {
        let err = min_position - articulation.position();
        let dvel = articulation_velocity[dof_id] + ext_vels[link.assembly_id + dof_id];

        if err >= Real::zero() {
            is_min_constraint_active = true;
            DVectorSliceMut::from_slice(&mut jacobians[*ground_j_id..], ndofs).fill(Real::zero());
            jacobians[*ground_j_id + link.assembly_id + dof_id] = Real::one();

            let wj_id = *ground_j_id + ndofs;
            multibody.inv_mass_mul_unit_articulation_force(
                link,
                dof_id,
                Real::one(),
                &mut jacobians[wj_id..],
            );

            let inv_r = jacobians[wj_id + link.assembly_id + dof_id]; // = J^t * M^-1 J

            let impulse_id = link.impulse_id + dof_id * 3 + 1;
            let constraint = UnilateralGroundConstraint {
                impulse: impulses[impulse_id] * parameters.warmstart_coeff,
                r: Real::one() / inv_r,
                rhs: dvel,
                impulse_id,
                assembly_id,
                j_id: *ground_j_id,
                wj_id: *ground_j_id + ndofs,
                ndofs,
            };

            constraints.velocity.unilateral_ground.push(constraint);
            *ground_j_id += 2 * ndofs;
        }
    }

    if let Some(max_position) = articulation.max_position() {
        let err = -(max_position - articulation.position());
        let dvel = -articulation_velocity[dof_id] - ext_vels[link.assembly_id + dof_id];

        if err >= Real::zero() {
            DVectorSliceMut::from_slice(&mut jacobians[*ground_j_id..], ndofs).fill(Real::zero());
            jacobians[*ground_j_id + link.assembly_id + dof_id] = -Real::one();
            let wj_id = *ground_j_id + ndofs;

            if is_min_constraint_active {
                // This jacobian is simply the negation of the first one.
                for i in 0..ndofs {
                    jacobians[wj_id + i] = -jacobians[*ground_j_id - ndofs + i];
                }
            } else {
                multibody.inv_mass_mul_unit_articulation_force(
                    link,
                    dof_id,
                    -Real::one(),
                    &mut jacobians[wj_id..],
                );
            }

            let inv_r = -jacobians[wj_id + link.assembly_id + dof_id]; // = J^t * M^-1 J

            let impulse_id = link.impulse_id + dof_id * 3 + 2;
            let constraint = UnilateralGroundConstraint {
                impulse: impulses[impulse_id] * parameters.warmstart_coeff,
                r: Real::one() / inv_r,
                rhs: dvel,
                impulse_id,
                assembly_id,
                j_id: *ground_j_id,
                wj_id: *ground_j_id + ndofs,
                ndofs,
            };

            constraints.velocity.unilateral_ground.push(constraint);
            *ground_j_id += 2 * ndofs;
        }
    }
}

/// Initializes and generate the position constraints applicable to the multibody links attached
/// to this articulation.
pub fn unit_articulation_position_constraint<Real: RealField, J: UnitArticulation>(
    articulation: &J,
    multibody: &Multibody<Real>,
    link: &MultibodyLink<Real>,
    handle: BodyPartHandle<()>,
    dof_id: usize,
    is_angular: bool,
    jacobians: &mut [Real],
) -> Option<GenericNonlinearConstraint<Real, ()>> {
    let mut sign = Real::one();
    let mut rhs = None;

    if let Some(min_position) = articulation.min_position() {
        let err = min_position - articulation.position();
        if err > Real::zero() {
            rhs = Some(-err);
        }
    }

    if rhs.is_none() {
        if let Some(max_position) = articulation.max_position() {
            let err = -(max_position - articulation.position());
            if err > Real::zero() {
                rhs = Some(-err);
                sign = -Real::one();
            }
        }
    }

    if let Some(rhs) = rhs {
        let ndofs = multibody.ndofs();

        multibody.inv_mass_mul_unit_articulation_force(link, dof_id, sign, jacobians);

        let inv_r = sign * jacobians[link.assembly_id + dof_id]; // = J^t * M^-1 J

        return Some(GenericNonlinearConstraint::new(
            handle,
            None,
            is_angular,
            ndofs,
            0,
            0,
            0,
            rhs,
            Real::one() / inv_r,
        ));
    }

    None
}
*/
