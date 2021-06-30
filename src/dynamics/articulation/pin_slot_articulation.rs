use na::{DVectorSliceMut, Isometry3, Unit, Vector3};

use crate::dynamics::{
    Articulation, IntegrationParameters, PrismaticArticulation, RevoluteArticulation,
    RigidBodyVelocity,
};
use crate::math::{JacobianSliceMut, Real};

/// A articulation that allows one translational and one rotational degrees of freedom.
///
/// Both are not required to be along the same direction.
#[derive(Copy, Clone, Debug)]
pub struct PinSlotArticulation {
    prism: PrismaticArticulation,
    revo: RevoluteArticulation,
}

impl PinSlotArticulation {
    /// Create a new pin-slot articulation with axes expressed in the local coordinate frame of the attached bodies, and
    /// with initial linear position and angle.
    pub fn new(
        axis_v: Unit<Vector3<Real>>,
        axis_w: Unit<Vector3<Real>>,
        position: Real,
        angle: Real,
    ) -> Self {
        let prism = PrismaticArticulation::new(axis_v, position);
        let revo = RevoluteArticulation::new(axis_w, angle);

        PinSlotArticulation { prism, revo }
    }

    /// The linear displacement.
    pub fn offset(&self) -> Real {
        self.prism.offset()
    }

    /// The angular displacement.
    pub fn angle(&self) -> Real {
        self.revo.angle()
    }
}

impl Articulation for PinSlotArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        2
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector3<Real>,
        body_shift: &Vector3<Real>,
    ) -> Isometry3<Real> {
        self.prism.translation() * self.revo.body_to_parent(parent_shift, body_shift)
    }

    fn update_jacobians(&mut self, body_shift: &Vector3<Real>, vels: &[Real]) {
        self.prism.update_jacobians(body_shift, vels);
        self.revo.update_jacobians(body_shift, &[vels[1]]);
    }

    fn jacobian(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        self.prism.jacobian(transform, &mut out.columns_mut(0, 1));
        self.revo.jacobian(transform, &mut out.columns_mut(1, 1));
    }

    fn jacobian_dot(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        self.prism
            .jacobian_dot(transform, &mut out.columns_mut(0, 1));
        self.revo
            .jacobian_dot(transform, &mut out.columns_mut(1, 1));
    }

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        transform: &Isometry3<Real>,
        vels: &[Real],
        out: &mut JacobianSliceMut<Real>,
    ) {
        self.prism.jacobian_dot_veldiff_mul_coordinates(
            transform,
            vels,
            &mut out.columns_mut(0, 1),
        );
        self.revo.jacobian_dot_veldiff_mul_coordinates(
            transform,
            &[vels[1]],
            &mut out.columns_mut(1, 1),
        );
    }

    fn jacobian_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        self.prism.jacobian_mul_coordinates(vels) + self.revo.jacobian_mul_coordinates(&[vels[1]])
    }

    fn jacobian_dot_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        // NOTE: The following is zero.
        // self.prism.jacobian_dot_mul_coordinates(vels) +
        self.revo.jacobian_dot_mul_coordinates(&[vels[1]])
    }

    fn default_damping(&self, out: &mut DVectorSliceMut<Real>) {
        self.prism.default_damping(&mut out.rows_mut(0, 1));
        self.revo.default_damping(&mut out.rows_mut(1, 1));
    }

    fn integrate(&mut self, dt: Real, vels: &[Real]) {
        self.prism.integrate(dt, vels);
        self.revo.integrate(dt, &[vels[1]]);
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        self.prism.apply_displacement(disp);
        self.revo.apply_displacement(&[disp[1]]);
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }

    /*
    fn num_velocity_constraints(&self) -> usize {
        self.prism.num_velocity_constraints() + self.revo.num_velocity_constraints()
    }

    fn velocity_constraints(
        &self,
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
        self.prism.velocity_constraints(
            parameters,
            multibody,
            link,
            assembly_id,
            dof_id,
            ext_vels,
            ground_j_id,
            jacobians,
            constraints,
        );
        self.revo.velocity_constraints(
            parameters,
            multibody,
            link,
            assembly_id,
            dof_id + 1,
            ext_vels,
            ground_j_id,
            jacobians,
            constraints,
        );
    }

    fn num_position_constraints(&self) -> usize {
        // NOTE: we don't test if constraints exist to simplify indexing.
        2
    }

    fn position_constraint(
        &self,
        i: usize,
        multibody: &Multibody<Real>,
        link: &MultibodyLink<Real>,
        handle: BodyPartHandle<()>,
        dof_id: usize,
        jacobians: &mut [Real],
    ) -> Option<GenericNonlinearConstraint<Real, ()>> {
        if i == 0 {
            self.prism
                .position_constraint(0, multibody, link, handle, dof_id, jacobians)
        } else {
            self.revo
                .position_constraint(0, multibody, link, handle, dof_id + 1, jacobians)
        }
    }
     */
}

prismatic_motor_limit_methods!(PinSlotArticulation, prism);
revolute_motor_limit_methods!(PinSlotArticulation, revo);
