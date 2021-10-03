use crate::data::graph::NodeIndex;
use crate::data::{Arena, Coarena};
use crate::dynamics::{Articulation, Multibody, RigidBodyHandle};
use crate::geometry::{InteractionGraph, RigidBodyGraphIndex, TemporaryInteractionIndex};
use crate::math::{Real, Vector};
use crate::parry::partitioning::IndexedData;
use std::ops::Index;

/// The unique handle of an articulation added to a `ArticulationSet`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct ArticulationHandle(pub crate::data::arena::Index);

/// The temporary index of a multibody added to a `ArticulationSet`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct MultibodyIndex(pub crate::data::arena::Index);

impl ArticulationHandle {
    /// Converts this handle into its (index, generation) components.
    pub fn into_raw_parts(self) -> (u32, u32) {
        self.0.into_raw_parts()
    }

    /// Reconstructs an handle from its (index, generation) components.
    pub fn from_raw_parts(id: u32, generation: u32) -> Self {
        Self(crate::data::arena::Index::from_raw_parts(id, generation))
    }

    /// An always-invalid rigid-body handle.
    pub fn invalid() -> Self {
        Self(crate::data::arena::Index::from_raw_parts(
            crate::INVALID_U32,
            crate::INVALID_U32,
        ))
    }
}

impl Default for ArticulationHandle {
    fn default() -> Self {
        Self::invalid()
    }
}

impl IndexedData for ArticulationHandle {
    fn default() -> Self {
        Self(IndexedData::default())
    }
    fn index(&self) -> usize {
        self.0.index()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArticulationLink {
    pub graph_id: RigidBodyGraphIndex,
    pub multibody: MultibodyIndex,
    pub id: usize,
}

impl Default for ArticulationLink {
    fn default() -> Self {
        Self {
            graph_id: RigidBodyGraphIndex::new(crate::INVALID_U32),
            multibody: MultibodyIndex(crate::data::arena::Index::from_raw_parts(
                crate::INVALID_U32,
                crate::INVALID_U32,
            )),
            id: 0,
        }
    }
}

/// A set of rigid bodies that can be handled by a physics pipeline.
pub struct ArticulationSet {
    pub(crate) multibodies: Arena<Multibody>, // NOTE: a Slab would be sufficient.
    pub(crate) rb2mb: Coarena<ArticulationLink>,
    // NOTE: this is mostly for the island extraction. So perhaps we won’t need
    //       that any more in the future when we improve our island builder.
    pub(crate) connectivity_graph: InteractionGraph<RigidBodyHandle, ()>,
}

impl ArticulationSet {
    /// Create a new empty set of multibodies.
    pub fn new() -> Self {
        Self {
            multibodies: Arena::new(),
            rb2mb: Coarena::new(),
            connectivity_graph: InteractionGraph::new(),
        }
    }

    /// Inserts a new articulation into this set.
    pub fn insert(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        articulation: impl Articulation,
        parent_shift: Vector<Real>,
        body_shift: Vector<Real>,
    ) -> Option<ArticulationHandle> {
        let link1 = self.rb2mb.get(body1.0).copied().unwrap_or_else(|| {
            let mb_handle = self.multibodies.insert(Multibody::with_root(body1));
            ArticulationLink {
                graph_id: self.connectivity_graph.graph.add_node(body1),
                multibody: MultibodyIndex(mb_handle),
                id: 0,
            }
        });

        let link2 = self.rb2mb.get(body2.0).copied().unwrap_or_else(|| {
            let mb_handle = self.multibodies.insert(Multibody::with_root(body2));
            ArticulationLink {
                graph_id: self.connectivity_graph.graph.add_node(body2),
                multibody: MultibodyIndex(mb_handle),
                id: 0,
            }
        });

        if link1.multibody == link2.multibody || link2.id != 0 {
            // This would introduce an invalid configuration.
            return None;
        }

        self.connectivity_graph
            .graph
            .add_edge(link1.graph_id, link2.graph_id, ());
        self.rb2mb.insert(body1.0, link1);
        self.rb2mb.insert(body2.0, link2);

        let mb2 = self.multibodies.remove(link2.multibody.0).unwrap();
        let multibody1 = &mut self.multibodies[link1.multibody.0];

        for mb_link2 in mb2.links() {
            let link = self.rb2mb.get_mut(mb_link2.rigid_body.0).unwrap();
            link.multibody = link1.multibody;
            link.id += multibody1.num_links();
        }

        multibody1.append(
            mb2,
            link1.id,
            Box::new(articulation),
            parent_shift,
            body_shift,
        );

        println!(
            "Articulation added, num multibodies: {}",
            self.multibodies.len()
        );

        // Because each rigid-body can only have one parent link,
        // we can use the second rigid-body’s handle as the articulation’s
        // handle.
        Some(ArticulationHandle(body2.0))
    }

    /// Returns the link of this multibody attached to the given rigid-body.
    ///
    /// Returns `None` if `rb` isn’t part of any rigid-body.
    pub fn rigid_body_link(&self, rb: RigidBodyHandle) -> Option<&ArticulationLink> {
        self.rb2mb.get(rb.0)
    }

    /// Gets a reference to a multibody, based on its temporary index.
    pub fn get_multibody(&self, index: MultibodyIndex) -> Option<&Multibody> {
        self.multibodies.get(index.0)
    }

    /// Gets a mutable reference to a multibody, based on its temporary index.
    ///
    /// This method will bypass any modification-detection automatically done by the
    /// `ArticulationSet`.
    pub fn get_multibody_mut_internal(&mut self, index: MultibodyIndex) -> Option<&mut Multibody> {
        self.multibodies.get_mut(index.0)
    }

    /// Gets a mutable reference to the multibody identified by its `handle`.
    pub fn get(&mut self, handle: ArticulationHandle) -> Option<(&Multibody, usize)> {
        let link = self.rb2mb.get(handle.0)?;
        let multibody = self.multibodies.get(link.multibody.0)?;
        Some((multibody, link.id))
    }

    /// Gets a mutable reference to the multibody identified by its `handle`.
    pub fn get_mut_internal(
        &mut self,
        handle: ArticulationHandle,
    ) -> Option<(&mut Multibody, usize)> {
        let link = self.rb2mb.get(handle.0)?;
        let multibody = self.multibodies.get_mut(link.multibody.0)?;
        Some((multibody, link.id))
    }

    /// Iterate through the handles of all the rigid-bodies attached to this rigid-body
    /// by an articulation.
    pub fn attached_bodies<'a>(
        &'a self,
        body: RigidBodyHandle,
    ) -> impl Iterator<Item = RigidBodyHandle> + 'a {
        self.rb2mb
            .get(body.0)
            .into_iter()
            .flat_map(move |id| self.connectivity_graph.interactions_with(id.graph_id))
            .map(move |inter| crate::utils::select_other((inter.0, inter.1), body))
    }
}

impl Index<MultibodyIndex> for ArticulationSet {
    type Output = Multibody;

    fn index(&self, index: MultibodyIndex) -> &Multibody {
        &self.multibodies[index.0]
    }
}

// impl Index<ArticulationHandle> for ArticulationSet {
//     type Output = Multibody;
//
//     fn index(&self, index: ArticulationHandle) -> &Multibody {
//         &self.multibodies[index.0]
//     }
// }
