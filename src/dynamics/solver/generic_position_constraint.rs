use crate::data::{BundleSet, ComponentSet, ComponentSetMut};
use crate::dynamics::solver::{PositionConstraint, PositionGroundConstraint};
use crate::dynamics::{
    ArticulationSet, IntegrationParameters, MultibodyIndex, RigidBodyIds, RigidBodyMassProps,
    RigidBodyPosition,
};
use crate::geometry::ContactManifold;
use crate::math::{Isometry, Point, Real, Rotation, Translation, Vector, MAX_MANIFOLD_POINTS};
use crate::utils::{WAngularInertia, WCross, WDot};
use na::DVector;

pub(crate) enum AnyGenericPositionConstraint {
    NonGroupedGround(GenericPositionGroundConstraint),
    NonGroupedNonGround(GenericPositionConstraint),
}

impl AnyGenericPositionConstraint {
    pub fn solve<Bodies>(
        &self,
        params: &IntegrationParameters,
        multibodies: &mut ArticulationSet,
        bodies: &mut Bodies,
        positions: &mut [Isometry<Real>],
        jacobians: &mut DVector<Real>,
    ) where
        Bodies: ComponentSetMut<RigidBodyMassProps> + ComponentSetMut<RigidBodyPosition>,
    {
        match self {
            AnyGenericPositionConstraint::NonGroupedGround(c) => {
                c.solve(params, multibodies, bodies, jacobians)
            }
            AnyGenericPositionConstraint::NonGroupedNonGround(c) => {
                c.solve(params, multibodies, bodies, positions, jacobians)
            }
        }
    }
}

pub(crate) struct GenericPositionConstraint {
    // We just build the generic constraint on top of the position constraint,
    // adding some information we can use in the generic case.
    pub position_constraint: PositionConstraint,
    pub multibody1: Option<MultibodyIndex>,
    pub multibody2: Option<MultibodyIndex>,
}

impl GenericPositionConstraint {
    pub fn generate<Bodies>(
        params: &IntegrationParameters,
        manifold: &ContactManifold,
        bodies: &Bodies,
        multibodies: &ArticulationSet,
        out_constraints: &mut Vec<AnyGenericPositionConstraint>,
        push: bool,
    ) where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        let handle1 = manifold.data.rigid_body1.unwrap();
        let handle2 = manifold.data.rigid_body2.unwrap();

        let ids1: &RigidBodyIds = bodies.index(handle1.0);
        let ids2: &RigidBodyIds = bodies.index(handle2.0);
        let poss1: &RigidBodyPosition = bodies.index(handle1.0);
        let poss2: &RigidBodyPosition = bodies.index(handle2.0);
        let mprops1: &RigidBodyMassProps = bodies.index(handle1.0);
        let mprops2: &RigidBodyMassProps = bodies.index(handle2.0);

        for (l, manifold_points) in manifold
            .data
            .solver_contacts
            .chunks(MAX_MANIFOLD_POINTS)
            .enumerate()
        {
            let mut local_p1 = [Point::origin(); MAX_MANIFOLD_POINTS];
            let mut local_p2 = [Point::origin(); MAX_MANIFOLD_POINTS];
            let mut dists = [0.0; MAX_MANIFOLD_POINTS];

            for l in 0..manifold_points.len() {
                local_p1[l] = poss1
                    .position
                    .inverse_transform_point(&manifold_points[l].point);
                local_p2[l] = poss2
                    .position
                    .inverse_transform_point(&manifold_points[l].point);
                dists[l] = manifold_points[l].dist;
            }

            let mb_link1 = multibodies.rigid_body_link(handle1);
            let mb_link2 = multibodies.rigid_body_link(handle2);

            let position_constraint = PositionConstraint {
                rb1: mb_link1.map(|l| l.id).unwrap_or(ids1.active_set_offset),
                rb2: mb_link2.map(|l| l.id).unwrap_or(ids2.active_set_offset),
                local_p1,
                local_p2,
                local_n1: poss1
                    .position
                    .inverse_transform_vector(&manifold.data.normal),
                dists,
                im1: mprops1.effective_inv_mass,
                im2: mprops2.effective_inv_mass,
                ii1: mprops1.effective_world_inv_inertia_sqrt.squared(),
                ii2: mprops2.effective_world_inv_inertia_sqrt.squared(),
                num_contacts: manifold_points.len() as u8,
                erp: params.erp,
                max_linear_correction: params.max_linear_correction,
            };
            let generic_constraint = GenericPositionConstraint {
                position_constraint,
                multibody1: mb_link1.map(|l| l.multibody),
                multibody2: mb_link2.map(|l| l.multibody),
            };

            if push {
                out_constraints.push(AnyGenericPositionConstraint::NonGroupedNonGround(
                    generic_constraint,
                ));
            } else {
                out_constraints[manifold.data.constraint_index + l] =
                    AnyGenericPositionConstraint::NonGroupedNonGround(generic_constraint);
            }
        }
    }

    pub fn solve<Bodies>(
        &self,
        params: &IntegrationParameters,
        multibodies: &mut ArticulationSet,
        bodies: &mut Bodies,
        positions: &mut [Isometry<Real>],
        jacobians: &mut DVector<Real>,
    ) where
        Bodies: ComponentSetMut<RigidBodyMassProps> + ComponentSetMut<RigidBodyPosition>,
    {
        let mut pos1 = if let Some(mb) = self.multibody1.map(|h| &multibodies[h]) {
            let handle = mb
                .link(self.position_constraint.rb1)
                .unwrap()
                .rigid_body_handle();
            let rb_pos: &RigidBodyPosition = bodies.index(handle.0);
            rb_pos.position
        } else {
            positions[self.position_constraint.rb1 as usize]
        };

        let mut pos2 = if let Some(mb) = self.multibody2.map(|h| &multibodies[h]) {
            let handle = mb
                .link(self.position_constraint.rb2)
                .unwrap()
                .rigid_body_handle();
            let rb_pos: &RigidBodyPosition = bodies.index(handle.0);
            rb_pos.position
        } else {
            positions[self.position_constraint.rb2 as usize]
        };

        let allowed_err = params.allowed_linear_error;

        for k in 0..self.position_constraint.num_contacts as usize {
            let target_dist = -self.position_constraint.dists[k] - allowed_err;
            let n1 = pos1 * self.position_constraint.local_n1;
            let p1 = pos1 * self.position_constraint.local_p1[k];
            let p2 = pos2 * self.position_constraint.local_p2[k];
            let dpos = p2 - p1;
            let dist = dpos.dot(&n1);

            if dist < target_dist {
                let p1 = p2 - n1 * dist;
                let err = ((dist - target_dist) * self.position_constraint.erp)
                    .max(-self.position_constraint.max_linear_correction);
                let dp1 = p1.coords - pos1.translation.vector;
                let dp2 = p2.coords - pos2.translation.vector;

                let gcross1 = dp1.gcross(n1);
                let gcross2 = -dp2.gcross(n1);

                let mut j_id1 = 0;

                let mut ii_gcross1 = Vector::zeros();
                let inv_r1;

                if let Some(multibody) = self.multibody1.map(|h| &multibodies[h]) {
                    inv_r1 = multibody.fill_jacobians(
                        self.position_constraint.rb1,
                        n1,
                        gcross1,
                        &mut j_id1,
                        jacobians,
                    )
                } else {
                    ii_gcross1 = self.position_constraint.ii1.transform_vector(gcross1);
                    inv_r1 = self.position_constraint.im1 + gcross1.gdot(ii_gcross1);
                };

                let mut j_id2 = j_id1;
                let mut ii_gcross2 = Vector::zeros();
                let inv_r2;

                if let Some(multibody) = self.multibody2.map(|h| &multibodies[h]) {
                    inv_r2 = multibody.fill_jacobians(
                        self.position_constraint.rb2,
                        -n1,
                        gcross2,
                        &mut j_id2,
                        jacobians,
                    );
                } else {
                    ii_gcross2 = self.position_constraint.ii2.transform_vector(gcross2);
                    inv_r2 = self.position_constraint.im2 + gcross2.gdot(ii_gcross2);
                };

                // Compute impulse.
                let impulse = err / (inv_r1 + inv_r2);

                // Apply impulse.
                if let Some(mb_handle1) = self.multibody1 {
                    let multibody1 = multibodies.get_multibody_mut_internal(mb_handle1).unwrap();
                    let ndofs = multibody1.ndofs();
                    let mut displacements = jacobians.rows_mut(j_id1 - ndofs, ndofs);
                    displacements *= impulse;
                    multibody1.apply_displacements(displacements.as_slice());
                    multibody1.forward_kinematics(bodies);
                } else {
                    let tra1 = Translation::from(n1 * (impulse * self.position_constraint.im1));
                    let rot1 = Rotation::new(ii_gcross1 * impulse);
                    pos1 = Isometry::from_parts(tra1 * pos1.translation, rot1 * pos1.rotation);
                }

                if let Some(mb_handle2) = self.multibody2 {
                    let multibody2 = multibodies.get_multibody_mut_internal(mb_handle2).unwrap();
                    let ndofs = multibody2.ndofs();
                    let mut displacements = jacobians.rows_mut(j_id2 - ndofs, ndofs);
                    displacements *= impulse;
                    multibody2.apply_displacements(displacements.as_slice());
                    multibody2.forward_kinematics(bodies);
                } else {
                    let tra2 = Translation::from(n1 * (-impulse * self.position_constraint.im2));
                    let rot2 = Rotation::new(ii_gcross2 * impulse);
                    pos2 = Isometry::from_parts(tra2 * pos2.translation, rot2 * pos2.rotation);
                }
            }
        }

        if self.multibody1.is_none() {
            positions[self.position_constraint.rb1] = pos1;
        }

        if self.multibody2.is_none() {
            positions[self.position_constraint.rb2] = pos2;
        }
    }
}

pub(crate) struct GenericPositionGroundConstraint {
    // We just build the generic constraint on top of the position constraint,
    // adding some information we can use in the generic case.
    pub position_constraint: PositionGroundConstraint,
    pub multibody2: MultibodyIndex,
}

impl GenericPositionGroundConstraint {
    pub fn generate<Bodies>(
        params: &IntegrationParameters,
        manifold: &ContactManifold,
        bodies: &Bodies,
        multibodies: &ArticulationSet,
        out_constraints: &mut Vec<AnyGenericPositionConstraint>,
        push: bool,
    ) where
        Bodies: ComponentSet<RigidBodyPosition>
            + ComponentSet<RigidBodyMassProps>
            + ComponentSet<RigidBodyIds>,
    {
        let flip = manifold.data.relative_dominance < 0;

        let (handle2, n1) = if flip {
            (manifold.data.rigid_body1.unwrap(), -manifold.data.normal)
        } else {
            (manifold.data.rigid_body2.unwrap(), manifold.data.normal)
        };

        let (poss2, mprops2): (&RigidBodyPosition, &RigidBodyMassProps) =
            bodies.index_bundle(handle2.0);

        for (l, manifold_contacts) in manifold
            .data
            .solver_contacts
            .chunks(MAX_MANIFOLD_POINTS)
            .enumerate()
        {
            let mut p1 = [Point::origin(); MAX_MANIFOLD_POINTS];
            let mut local_p2 = [Point::origin(); MAX_MANIFOLD_POINTS];
            let mut dists = [0.0; MAX_MANIFOLD_POINTS];

            for k in 0..manifold_contacts.len() {
                p1[k] = manifold_contacts[k].point;
                local_p2[k] = poss2
                    .position
                    .inverse_transform_point(&manifold_contacts[k].point);
                dists[k] = manifold_contacts[k].dist;
            }

            let mb_link2 = multibodies.rigid_body_link(handle2).unwrap();

            let position_constraint = PositionGroundConstraint {
                rb2: mb_link2.id,
                p1,
                local_p2,
                n1,
                dists,
                im2: mprops2.effective_inv_mass,
                ii2: mprops2.effective_world_inv_inertia_sqrt.squared(),
                num_contacts: manifold_contacts.len() as u8,
                erp: params.erp,
                max_linear_correction: params.max_linear_correction,
            };
            let generic_constraint = GenericPositionGroundConstraint {
                position_constraint,
                multibody2: mb_link2.multibody,
            };

            if push {
                out_constraints.push(AnyGenericPositionConstraint::NonGroupedGround(
                    generic_constraint,
                ));
            } else {
                out_constraints[manifold.data.constraint_index + l] =
                    AnyGenericPositionConstraint::NonGroupedGround(generic_constraint);
            }
        }
    }

    pub fn solve<Bodies>(
        &self,
        params: &IntegrationParameters,
        multibodies: &mut ArticulationSet,
        bodies: &mut Bodies,
        jacobians: &mut DVector<Real>,
    ) where
        Bodies: ComponentSetMut<RigidBodyMassProps> + ComponentSetMut<RigidBodyPosition>,
    {
        let multibody2 = multibodies
            .get_multibody_mut_internal(self.multibody2)
            .unwrap();
        let handle = multibody2
            .link(self.position_constraint.rb2)
            .unwrap()
            .rigid_body_handle();
        let rb_pos: &RigidBodyPosition = bodies.index(handle.0);
        let pos2 = rb_pos.position;

        let allowed_err = params.allowed_linear_error;

        for k in 0..self.position_constraint.num_contacts as usize {
            let target_dist = -self.position_constraint.dists[k] - allowed_err;
            let p2 = pos2 * self.position_constraint.local_p2[k];
            let dpos = p2 - self.position_constraint.p1[k];
            let dist = dpos.dot(&self.position_constraint.n1);

            if dist < target_dist {
                let err = ((dist - target_dist) * self.position_constraint.erp)
                    .max(-self.position_constraint.max_linear_correction);
                let dp2 = p2.coords - pos2.translation.vector;
                let gcross2 = -dp2.gcross(self.position_constraint.n1);

                let mut j_id2 = 0;

                let inv_r2 = multibody2.fill_jacobians(
                    self.position_constraint.rb2,
                    -self.position_constraint.n1,
                    gcross2,
                    &mut j_id2,
                    jacobians,
                );

                // Compute impulse.
                let impulse = err / inv_r2;

                // Apply impulse.
                let ndofs = multibody2.ndofs();
                let mut displacements = jacobians.rows_mut(j_id2 - ndofs, ndofs);
                displacements *= impulse;
                multibody2.apply_displacements(displacements.as_slice());
                multibody2.forward_kinematics(bodies);
            }
        }
    }
}
