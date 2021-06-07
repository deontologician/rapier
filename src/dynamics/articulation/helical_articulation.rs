use na::{self, DVectorSliceMut, Isometry3, Translation3, Unit, Vector3};

use crate::dynamics::{
    Articulation, ArticulationMotor, IntegrationParameters, RevoluteArticulation,
    RigidBodyVelocity, UnitArticulation,
};
use crate::math::{JacobianSliceMut, Real};

/// A articulation that allows one degree of freedom between two multibody links.
///
/// The degree of freedom is the combination of a rotation and a translation along the same axis.
/// Both rotational and translational motions are coupled to generate a screw motion.
#[derive(Copy, Clone, Debug)]
pub struct HelicalArticulation {
    revo: RevoluteArticulation,
    pitch: Real,
}

impl HelicalArticulation {
    /// Create an helical articulation with the given axis and initial angle.
    ///
    /// The `pitch` controls how much translation is generated for how much rotation.
    /// In particular, the translational displacement along `axis` is given by `angle * pitch`.
    pub fn new(axis: Unit<Vector3<Real>>, pitch: Real, angle: Real) -> Self {
        HelicalArticulation {
            revo: RevoluteArticulation::new(axis, angle),
            pitch: pitch,
        }
    }

    /// The translational displacement along the articulation axis.
    pub fn offset(&self) -> Real {
        self.revo.angle() * self.pitch
    }

    /// The rotational displacement along the articulation axis.
    pub fn angle(&self) -> Real {
        self.revo.angle()
    }
}

impl Articulation for HelicalArticulation {
    #[inline]
    fn ndofs(&self) -> usize {
        1
    }

    fn body_to_parent(
        &self,
        parent_shift: &Vector3<Real>,
        body_shift: &Vector3<Real>,
    ) -> Isometry3<Real> {
        Translation3::from(self.revo.axis().as_ref() * self.revo.angle())
            * self.revo.body_to_parent(parent_shift, body_shift)
    }

    fn update_jacobians(&mut self, body_shift: &Vector3<Real>, vels: &[Real]) {
        self.revo.update_jacobians(body_shift, vels)
    }

    fn jacobian(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        let mut jac = *self.revo.local_jacobian();
        jac.linvel += self.revo.axis().as_ref() * self.pitch;
        out.copy_from(jac.transformed(transform).as_vector())
    }

    fn jacobian_dot(&self, transform: &Isometry3<Real>, out: &mut JacobianSliceMut<Real>) {
        self.revo.jacobian_dot(transform, out)
    }

    fn jacobian_dot_veldiff_mul_coordinates(
        &self,
        transform: &Isometry3<Real>,
        acc: &[Real],
        out: &mut JacobianSliceMut<Real>,
    ) {
        self.revo
            .jacobian_dot_veldiff_mul_coordinates(transform, acc, out)
    }

    fn jacobian_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        let mut jac = *self.revo.local_jacobian();
        jac.linvel += self.revo.axis().as_ref() * self.pitch;
        jac * vels[0]
    }

    fn jacobian_dot_mul_coordinates(&self, vels: &[Real]) -> RigidBodyVelocity {
        self.revo.jacobian_dot_mul_coordinates(vels)
    }

    fn default_damping(&self, out: &mut DVectorSliceMut<Real>) {
        out.fill(na::convert(0.1f64))
    }

    fn integrate(&mut self, parameters: &IntegrationParameters, vels: &[Real]) {
        self.revo.integrate(parameters, vels)
    }

    fn apply_displacement(&mut self, disp: &[Real]) {
        self.revo.apply_displacement(disp)
    }

    #[inline]
    fn clone(&self) -> Box<dyn Articulation> {
        Box::new(*self)
    }

    /*
    fn num_velocity_constraints(&self) -> usize {
        self.revo.num_velocity_constraints()
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
        // XXX: is this correct even though we don't have the same jacobian?
        self.revo.velocity_constraints(
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
    }

    fn num_position_constraints(&self) -> usize {
        // NOTE: we don't test if constraints exist to simplify indexing.
        1
    }

    fn position_constraint(
        &self,
        _: usize,
        multibody: &Multibody,
        link: &MultibodyLink,
        handle: BodyPartHandle<()>,
        dof_id: usize,
        jacobians: &mut [Real],
    ) -> Option<GenericNonlinearConstraint<Real, ()>> {
        // XXX: is this correct even though we don't have the same jacobian?
        self.revo
            .position_constraint(0, multibody, link, handle, dof_id, jacobians)
    }
     */
}

impl UnitArticulation for HelicalArticulation {
    fn position(&self) -> Real {
        self.revo.angle()
    }

    fn motor(&self) -> &ArticulationMotor<Real> {
        self.revo.motor()
    }

    fn min_position(&self) -> Option<Real> {
        self.revo.min_angle()
    }

    fn max_position(&self) -> Option<Real> {
        self.revo.max_angle()
    }
}

revolute_motor_limit_methods!(HelicalArticulation, revo);
