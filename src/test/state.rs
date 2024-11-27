use crate::*;
use core::{fmt::Debug, marker::PhantomData};
use imbl::OrdSet;
use proptest::prelude::*;
use proptest_state_machine::{prop_state_machine, ReferenceStateMachine, StateMachineTest};

prop_state_machine! {
    #[test]
    fn test_state_machine(sequential 1..20 => FlatBitSet);
}

pub struct SetStateMachine<K>(PhantomData<K>);

#[derive(Clone, Debug)]
pub enum Transition<K> {
    Insert(K),
    Remove(K),
    Toggle(K),
    Set(K, bool),
}

impl<K> ReferenceStateMachine for SetStateMachine<K>
where
    K: Key + Debug + Arbitrary,
{
    type State = (OrdSet<K>, bool);

    type Transition = Transition<K>;

    fn init_state() -> BoxedStrategy<Self::State> {
        (imbl::proptest::ord_set(any::<K>(), 0..20), Just(false)).boxed()
    }

    fn transitions((set, _): &Self::State) -> BoxedStrategy<Self::Transition> {
        let el = (0..set.len()).prop_map({
            let state = set.clone();
            move |n| *state.iter().nth(n).unwrap()
        });
        let ne = |n| if set.is_empty() { 0 } else { n };
        prop_oneof![
            1 => any::<K>().prop_map(Transition::Insert),
            ne(1) => el.clone().prop_map(Transition::Insert),
            1 => any::<K>().prop_map(Transition::Remove),
            ne(1) => el.clone().prop_map(Transition::Remove),
            1 => any::<K>().prop_map(Transition::Toggle),
            ne(1) => el.clone().prop_map(Transition::Toggle),
            1 => any::<(K, bool)>().prop_map(|(k, p)| Transition::Set(k, p)),
            ne(1) => (el.clone(), any::<bool>()).prop_map(|(k, p)| Transition::Set(k, p)),
        ]
        .boxed()
    }

    fn apply((mut set, _): Self::State, transition: &Self::Transition) -> Self::State {
        let old = match transition {
            Transition::Insert(k) => set.insert(*k).is_some(),
            Transition::Remove(k) => set.remove(k).is_some(),
            Transition::Toggle(k) => {
                let old = set.contains(k);
                if old {
                    set.remove(k);
                } else {
                    set.insert(*k);
                }
                old
            }
            Transition::Set(k, p) => {
                if *p {
                    set.insert(*k).is_some()
                } else {
                    set.remove(k).is_some()
                }
            }
        };
        (set, old)
    }
}

impl<K, C> StateMachineTest for FlatBitSet<K, C>
where
    K: Key + Arbitrary,
    C: Chunk,
{
    type SystemUnderTest = Self;

    type Reference = SetStateMachine<K>;

    fn init_test(
        (set, _): &<Self::Reference as ReferenceStateMachine>::State,
    ) -> Self::SystemUnderTest {
        set.iter().copied().collect()
    }

    fn apply(
        mut state: Self::SystemUnderTest,
        ref_state: &<Self::Reference as ReferenceStateMachine>::State,
        transition: <Self::Reference as ReferenceStateMachine>::Transition,
    ) -> Self::SystemUnderTest {
        let old = match transition {
            Transition::Insert(k) => !state.insert(k),
            Transition::Remove(k) => state.remove(k),
            Transition::Toggle(k) => state.toggle(k),
            Transition::Set(k, p) => state.set(k, p),
        };
        assert_eq!(ref_state.1, old);
        assert!(ref_state.0.iter().copied().eq(&state),);
        state
    }
}
