use gpurs_artifact::{Convergence, KernelEffects, Memory, Primitive, Proof};
use gpurs_rustc::{CrateDef, Instance, Item};

use super::{Analysis, Origin, Pointer, Region, State, Value, ValueData, state};
use crate::symbolic::{Atom, Polynomial, Predicate, Relation};

impl Analysis<'_, '_> {
    pub(super) fn primitive(
        &mut self,
        primitive: &Primitive,
        instance: Instance,
        arguments: &[Value],
        state: &mut State,
    ) -> Value {
        let item = self.sess.item(instance.def.def_id());
        if matches!(item, Some(Item::ScatterLoad | Item::ScatterStore))
            && let Some(Pointer::Memory(mut origin)) = arguments[0].pointer.clone()
        {
            origin.index.clone_from(&arguments[1].integer);
            origin.inner = false;
            self.access(&origin, item == Some(Item::ScatterStore), false, state);
        }
        match primitive.memory {
            Memory::Read | Memory::Write | Memory::Atomic => {
                if primitive.memory == Memory::Atomic
                    && matches!(
                        arguments.first().and_then(|value| value.pointer.as_ref()),
                        Some(Pointer::Slot(..))
                    )
                {
                    self.sess.error_help(
                        self.span,
                        "native GPU atomics require global or block-shared storage",
                        "use a device buffer or shared_array instead of thread-local storage",
                    );
                }
                if let Some(Pointer::Memory(origin)) = arguments
                    .first()
                    .and_then(|argument| argument.pointer.clone())
                {
                    if primitive.memory == Memory::Atomic {
                        self.access(&origin, false, true, state);
                    }
                    self.access(
                        &origin,
                        primitive.memory != Memory::Read,
                        primitive.memory == Memory::Atomic,
                        state,
                    );
                }
            }
            Memory::ReadReachable(argument) | Memory::BlockFenceReadReachable(argument) => {
                let mut regions = Vec::new();
                reachable(&arguments[argument], state, 3, &mut regions);
                for region in regions {
                    let origin = Origin {
                        region,
                        index: None,
                        inner: false,
                        path: Vec::new(),
                    };
                    self.access(&origin, false, false, state);
                }
            }
            Memory::PartitionRead | Memory::PartitionWrite => self.partition(
                instance,
                &arguments[0],
                primitive.memory == Memory::PartitionWrite,
                state,
            ),
            Memory::None | Memory::BlockFence | Memory::SharedAllocation | Memory::Encapsulated => {
            }
        }
        self.findings.effects |= primitive.effects() & KernelEffects::INDEX;
        self.findings.geometry |= primitive.proof != Proof::Index
            && (primitive.geometry()
                || primitive.convergence != Convergence::None
                || primitive.memory != Memory::None);
        if self.collect && self.quiet == 0 {
            let block = self.kernel_block();
            match primitive.convergence {
                Convergence::None => {}
                Convergence::Warp => {
                    self.findings
                        .collectives
                        .push((block, 1, self.span, instance));
                    let threads = (0..3).fold(Polynomial::constant(1), |product, axis| {
                        product
                            .multiply(&Polynomial::atom(Atom::BlockDimension(axis)))
                            .unwrap()
                    });
                    self.requirement(
                        Predicate::new(
                            Polynomial::remainder(threads, Polynomial::constant(32)),
                            Relation::Zero,
                        ),
                        "warp operations require complete warps",
                    );
                }
                Convergence::Block => {
                    self.findings
                        .collectives
                        .push((block, 0, self.span, instance));
                    if matches!(
                        primitive.memory,
                        Memory::BlockFence | Memory::BlockFenceReadReachable(_)
                    ) {
                        self.findings.barriers.insert(block);
                    }
                }
            }
        }
        let axes = |atom: fn(usize) -> Atom, variance| {
            Value::from(ValueData {
                fields: (0..3)
                    .map(|axis| Value::integer(Polynomial::atom(atom(axis)), variance))
                    .collect(),
                variance,
                ..ValueData::default()
            })
        };
        match primitive.proof {
            Proof::ThreadAxes => axes(Atom::Thread, 2),
            Proof::BlockAxes => axes(Atom::Block, 0),
            Proof::BlockDimensionAxes => axes(Atom::BlockDimension, 0),
            Proof::GridDimensionAxes => axes(Atom::GridDimension, 0),
            Proof::Index => Value::integer(Polynomial::atom(Atom::Index), 2),
            Proof::Lane => {
                if self.collect {
                    self.requirement(
                        Predicate::new(
                            Polynomial::remainder(
                                Polynomial::atom(Atom::BlockDimension(0)),
                                Polynomial::constant(32),
                            ),
                            Relation::Zero,
                        ),
                        "warp lane proofs require thread-x rows aligned to complete warps",
                    );
                }
                Value::integer(
                    Polynomial::remainder(
                        Polynomial::atom(Atom::Thread(0)),
                        Polynomial::constant(32),
                    ),
                    2,
                )
            }
            Proof::SharedRegion => {
                let length = instance.args().0[1]
                    .expect_const()
                    .eval_target_usize()
                    .unwrap();
                let pointer = Some(Pointer::Memory(Origin {
                    region: Region::Shared(self.kernel_block(), length),
                    index: Some(Polynomial::default()),
                    inner: false,
                    path: Vec::new(),
                }));
                Value::from(ValueData {
                    fields: vec![Value::from(ValueData {
                        pointer: pointer.clone(),
                        ..ValueData::default()
                    })],
                    pointer,
                    ..ValueData::default()
                })
            }
            Proof::BlockShape => {
                let threads = instance.args().0[0]
                    .expect_const()
                    .eval_target_usize()
                    .unwrap();
                if self.collect {
                    for (axis, count) in [(0, threads), (1, 1), (2, 1)] {
                        self.requirement(
                            Predicate::new(
                                Polynomial::atom(Atom::BlockDimension(axis))
                                    .offset(-(count as i128))
                                    .unwrap(),
                                Relation::Zero,
                            ),
                            "the block must have the shape its declaration requires",
                        );
                    }
                }
                Value::varying(0)
            }
            Proof::Varying => Value::varying(2),
            Proof::Uniform | Proof::Initialized => Value::varying(0),
        }
    }
}

/// The regions `value` may point into, following local pointers up to `depth` levels.
fn reachable(value: &Value, state: &State, depth: usize, regions: &mut Vec<Region>) {
    let region = match &value.pointer {
        Some(Pointer::Memory(origin)) => Some(origin.region.clone()),
        Some(Pointer::Slot(slot, path)) if depth > 0 => {
            reachable(
                &state::get(&state.slots[*slot], path),
                state,
                depth - 1,
                regions,
            );
            None
        }
        Some(_) => Some(Region::Unknown),
        None => None,
    };
    if let Some(region) = region
        && !regions.contains(&region)
    {
        regions.push(region);
    }
    for field in &value.fields {
        reachable(field, state, depth, regions);
    }
}
