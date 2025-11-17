import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

SUSPECT_NAME_DICTIONARY = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank', 'Ivy', 'Jack']

TIME_ATTRIBUTE_NAME = 'Time'
ATTRIBUTE_DOMAIN_MAP = {
    'Room': ['Kitchen', 'Library', 'Study', 'Conservatory', 'Ballroom'],
    'Weapon': ['Axe', 'Knife', 'Rope', 'Wrench', 'Pistol'],
    'Time': list(range(0, 24)),
    'Entry': ['Front door', 'Back door', 'Window', 'Secret passage'],
    'Motive': ['Owes money to the victim', 'Was seen arguing with the victim', 'Illicit relationship with the victim']
}

ATTRIBUTE_NAMES = list(ATTRIBUTE_DOMAIN_MAP.keys())
NUM_ATTRIBUTES = len(ATTRIBUTE_DOMAIN_MAP)


@dataclass
class PotentialWorlds:
    potential_attributes: Dict[str, Dict[str, Set[Any]]]
    potential_crime_attributes: Dict[str, Set[Any]]

    def apply_one(self, clue: 'Clue') -> bool:
        return clue.apply(self)

    def apply_many(self, clues: List['Clue']) -> bool:
        for clue in clues:
            if not clue.apply(self):
                return False
        return True

    def num_potential_murderers(self) -> int:
        count = 0
        for sname, sdoms in self.potential_attributes.items():
            fits = True
            for aname in ATTRIBUTE_NAMES:
                if not any(v in self.potential_crime_attributes[aname] for v in sdoms[aname]):
                    fits = False
                    break
            if fits:
                count += 1
        return count

    def get_total_domain_size(self) -> int:
        return sum(len(dom) for sdoms in self.potential_attributes.values() for dom in sdoms.values()) \
            + sum(len(dom) for dom in self.potential_crime_attributes.values())


@dataclass
class World:
    suspects_attributes: Dict[str, Dict[str, Any]]
    murderer_name: str
    crime_attributes: Dict[str, Any]
    suspect_names: List[str]

    def get_suspect_names(self):
        return list(self.suspects_attributes.keys())

    @staticmethod
    def generate_world(num_suspects: int):
        suspect_names = random.sample(SUSPECT_NAME_DICTIONARY, k=num_suspects)
        murderer_name = random.choice(suspect_names)
        suspects_attributes: Dict[str, Dict[str, Any]] = {}
        for sname in suspect_names:
            attrs = {an: random.choice(ATTRIBUTE_DOMAIN_MAP[an]) for an in ATTRIBUTE_NAMES}
            while attrs in suspects_attributes.values():
                attrs = {an: random.choice(ATTRIBUTE_DOMAIN_MAP[an]) for an in ATTRIBUTE_NAMES}
            suspects_attributes[sname] = attrs
        murderer_time = suspects_attributes[murderer_name][TIME_ATTRIBUTE_NAME]
        crime_attrs = {k: v for k, v in suspects_attributes[murderer_name].items()}
        possible_ct = [t for t in ATTRIBUTE_DOMAIN_MAP[TIME_ATTRIBUTE_NAME] if t >= murderer_time]
        crime_attrs[TIME_ATTRIBUTE_NAME] = random.choice(possible_ct) if possible_ct else murderer_time
        return World(suspects_attributes=suspects_attributes,
                     murderer_name=murderer_name,
                     crime_attributes=crime_attrs,
                     suspect_names=suspect_names)

    def __repr__(self):
        return f'World(suspects={self.suspects_attributes}, murderer_name={self.murderer_name}, crime={self.crime_attributes}, suspects_names={self.suspect_names})'


class Clue(ABC):
    @abstractmethod
    def apply(self, pw: PotentialWorlds) -> bool:
        """
        Modify the potential world based on this clue, or returns false if this would result in a contradiction
        :param potential_worlds:
        :return:
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """
        Get human readable representation of the clue
        :return:
        """
        pass


@dataclass
class UnaryPositive(Clue):
    suspect_name: str
    attribute_name: str
    value: Any

    def apply(self, pw: PotentialWorlds) -> bool:
        dom = pw.potential_attributes[self.suspect_name][self.attribute_name]
        if self.value not in dom:
            return False
        pw.potential_attributes[self.suspect_name][self.attribute_name] = {self.value}
        return True

    def describe(self):
        return f"{self.suspect_name} has {self.attribute_name} = {self.value}"

    def __repr__(self):
        return f'UnaryPositive(suspect_name={self.suspect_name}, attribute_name={self.attribute_name}, value={self.value})'


@dataclass
class UnaryNegative(Clue):
    suspect_name: str
    attribute_name: str
    value: Any

    def apply(self, pw: PotentialWorlds) -> bool:
        dom = pw.potential_attributes[self.suspect_name][self.attribute_name]
        if self.value in dom:
            dom.discard(self.value)
            if not dom:
                return False
        return True

    def describe(self):
        return f"{self.suspect_name} does not have {self.attribute_name} = {self.value}"

    def __repr__(self):
        return f'UnaryNegative(suspect_name={self.suspect_name}, attribute_name={self.attribute_name}, value={self.value})'


@dataclass
class CrimePositive(Clue):
    attribute_name: str
    value: Any

    def apply(self, pw: PotentialWorlds) -> bool:
        dom = pw.potential_crime_attributes[self.attribute_name]
        if self.value not in dom:
            return False
        pw.potential_crime_attributes[self.attribute_name] = {self.value}
        return True

    def describe(self):
        return f'The murder {self.attribute_name} was {self.value}'

    def __repr__(self):
        return f'CrimePositive(attribute_name={self.attribute_name}, value={self.value})'


@dataclass
class CrimeNegative(Clue):
    attribute_name: str
    value: Any

    def apply(self, pw: PotentialWorlds) -> bool:
        dom = pw.potential_crime_attributes[self.attribute_name]
        if self.value in dom:
            dom.discard(self.value)
            if not dom:
                return False
        return True

    def describe(self):
        return f'The murder {self.attribute_name} was NOT {self.value}'

    def __repr__(self):
        return f'CrimeNegative(attribute_name={self.attribute_name}, value={self.value})'


@dataclass
class TimeOrdering(Clue):
    suspect_name: str
    comparison: int  # 0 => crime_time <= suspect_time; 1 => crime_time >= suspect_time

    def apply(self, pw: PotentialWorlds) -> bool:
        sd = pw.potential_attributes[self.suspect_name][TIME_ATTRIBUTE_NAME]
        cd = pw.potential_crime_attributes[TIME_ATTRIBUTE_NAME]
        if not sd or not cd:
            return False
        if self.comparison == 0:
            new_sd = {t for t in sd if any(ct <= t for ct in cd)}
            new_cd = {ct for ct in cd if any(ct <= t for t in sd)}
        else:
            new_sd = {t for t in sd if any(ct >= t for ct in cd)}
            new_cd = {ct for ct in cd if any(ct >= t for t in sd)}
        if not new_sd or not new_cd:
            return False
        pw.potential_attributes[self.suspect_name][TIME_ATTRIBUTE_NAME] = new_sd
        pw.potential_crime_attributes[TIME_ATTRIBUTE_NAME] = new_cd
        return True

    def describe(self):
        return f'The murder happened {["before or on", "after or on"][self.comparison]} {self.suspect_name}\'s arrival'

    def __repr__(self):
        return f'TimeOrdering(suspect_name={self.suspect_name}, comparison={self.comparison})'


@dataclass
class IndirectPositive(Clue):
    attribute_name_1: str
    value_1: Any
    attribute_name_2: str
    value_2: Any

    def apply(self, pw: PotentialWorlds) -> bool:
        a1, v1 = self.attribute_name_1, self.value_1
        a2, v2 = self.attribute_name_2, self.value_2
        changed = True
        while changed:
            changed = False
            for sname, sdoms in pw.potential_attributes.items():
                dA, dB = sdoms[a1], sdoms[a2]
                if v1 in dA and v2 not in dB:
                    dA.discard(v1)
                    if not dA:
                        return False
                    changed = True
                if dA == {v1}:
                    if v2 not in dB:
                        return False
                    if dB != {v2}:
                        sdoms[a2] = {v2}
                        changed = True
        if not any(v1 in sdoms[a1] for sdoms in pw.potential_attributes.values()):
            return False
        return True

    def describe(self):
        return f'The individual(s) who had {self.attribute_name_1} of {self.value_1}, had {self.attribute_name_2} of {self.value_2}'

    def __repr__(self):
        return f'IndirectPositive(attribute_name_1={self.attribute_name_1}, value_1={self.value_1}, attribute_name_2={self.attribute_name_2}, value_2={self.value_2})'


@dataclass
class IndirectNegative(Clue):
    attribute_name_1: str
    value_1: Any
    attribute_name_2: str
    value_2: Any

    def apply(self, pw: PotentialWorlds) -> bool:
        a1, v1 = self.attribute_name_1, self.value_1
        a2, v2 = self.attribute_name_2, self.value_2
        changed = True
        while changed:
            changed = False
            for sname, sdoms in pw.potential_attributes.items():
                dA, dB = sdoms[a1], sdoms[a2]
                if dA == {v1} and v2 in dB:
                    dB.discard(v2)
                    if not dB:
                        return False
                    changed = True
                if dB == {v2} and v1 in dA:
                    dA.discard(v1)
                    if not dA:
                        return False
                    changed = True
        if not any(v1 in sdoms[a1] for sdoms in pw.potential_attributes.values()):
            return False
        return True

    def describe(self):
        return f'The individual(s) who had {self.attribute_name_1} of {self.value_1}, did not have {self.attribute_name_2} of {self.value_2}'

    def __repr__(self):
        return f'IndirectNegative(attribute_name_1={self.attribute_name_1}, value_1={self.value_1}, attribute_name_2={self.attribute_name_2}, value_2={self.value_2})'


@dataclass
class Puzzle:
    world: World
    clues: List[Clue]

    def get_potential_worlds(self):
        return get_potential_worlds(self.world, self.clues)

    def estimate_alpha_beta(self):
        num_unary = sum(1 for clue in self.clues if
                        isinstance(clue, (UnaryPositive, UnaryNegative, CrimePositive, CrimeNegative, TimeOrdering)))
        num_indirect_positive = sum(1 for clue in self.clues if isinstance(clue, IndirectPositive))
        num_indirect_negative = sum(1 for clue in self.clues if isinstance(clue, IndirectNegative))
        total = len(self.clues)
        alpha = num_unary / total if total > 0 else 0
        beta = num_indirect_positive / (num_indirect_positive + num_indirect_negative) if (
                                                                                                  num_indirect_positive + num_indirect_negative) > 0 else 0
        return alpha, beta

    def to_json(self) -> Dict[str, Any]:
        suspects_json = {
            sname: {
                aname: value
                for aname, value in attrs.items()
            }
            for sname, attrs in self.world.suspects_attributes.items()
        }

        clues_json = [
            {
                "type": clue.__class__.__name__,
                "repr": repr(clue),
                "text": clue.describe()
            }
            for clue in self.clues
        ]

        truth_json = {
            "murderer_name": self.world.murderer_name,
            "crime_attributes": self.world.crime_attributes
        }

        return {
            "suspects": suspects_json,
            "clues": clues_json,
            "hidden_truth": truth_json
        }

    # convenience string export
    def to_json_string(self, **kwargs) -> str:
        """Return a pretty JSON string."""
        return json.dumps(self.to_json(), indent=4, **kwargs)

    def __repr__(self):
        return f'Puzzle(world={self.world}, clues={self.clues})'


def generate_clues(world: World) -> List[Clue]:
    clues: List[Clue] = []
    for sname, attrs in world.suspects_attributes.items():
        for aname in ATTRIBUTE_NAMES:
            if aname == TIME_ATTRIBUTE_NAME: continue  # filter out time-based unary clues

            true_val = attrs[aname]
            for v in ATTRIBUTE_DOMAIN_MAP[aname]:
                if v == true_val:
                    clues.append(UnaryPositive(sname, aname, v))
                else:
                    clues.append(UnaryNegative(sname, aname, v))
    crime_attrs = world.crime_attributes
    for aname in ATTRIBUTE_NAMES:
        for v in ATTRIBUTE_DOMAIN_MAP[aname]:
            if v == crime_attrs[aname]:
                clues.append(CrimePositive(aname, v))
            else:
                clues.append(CrimeNegative(aname, v))

    crime_time = world.crime_attributes[TIME_ATTRIBUTE_NAME]
    for sname, attrs in world.suspects_attributes.items():
        st = attrs[TIME_ATTRIBUTE_NAME]
        if crime_time <= st:
            clues.append(TimeOrdering(sname, 0))
        if crime_time >= st:
            clues.append(TimeOrdering(sname, 1))

    for i in range(NUM_ATTRIBUTES):
        for j in range(NUM_ATTRIBUTES):
            if i == j:
                continue
            a1, a2 = ATTRIBUTE_NAMES[i], ATTRIBUTE_NAMES[j]
            groups = {}
            for attrs in world.suspects_attributes.values():
                v1 = attrs[a1]
                groups.setdefault(v1, []).append(attrs)
            for v1, group in groups.items():
                b_vals = {attrs[a2] for attrs in group}
                if len(b_vals) == 1:
                    v2 = next(iter(b_vals))
                    clues.append(IndirectPositive(a1, v1, a2, v2))
            for v1, group in groups.items():
                observed_b = {attrs[a2] for attrs in group}
                for v2 in ATTRIBUTE_DOMAIN_MAP[a2]:
                    if v2 not in observed_b:
                        clues.append(IndirectNegative(a1, v1, a2, v2))
    return clues


def _enforce_crime_after_murderer_time(world: World, pw: PotentialWorlds) -> bool:
    mname = world.murderer_name
    md = pw.potential_attributes[mname][TIME_ATTRIBUTE_NAME]
    cd = pw.potential_crime_attributes[TIME_ATTRIBUTE_NAME]
    if not md or not cd:
        return False
    new_cd = {ct for ct in cd if any(ct >= mt for mt in md)}
    new_md = {mt for mt in md if any(ct >= mt for ct in cd)}
    if not new_cd or not new_md:
        return False
    pw.potential_crime_attributes[TIME_ATTRIBUTE_NAME] = new_cd
    pw.potential_attributes[mname][TIME_ATTRIBUTE_NAME] = new_md
    return True


def get_potential_worlds(world: World, clues: List[Clue]) -> PotentialWorlds | None:
    pw = PotentialWorlds(
        potential_attributes={
            sname: {aname: set(ATTRIBUTE_DOMAIN_MAP[aname]) for aname in ATTRIBUTE_NAMES}
            for sname in world.suspect_names
        },
        potential_crime_attributes={aname: set(ATTRIBUTE_DOMAIN_MAP[aname]) for aname in ATTRIBUTE_NAMES}
    )
    after, before = 0, -1
    while after != before:
        before = pw.get_total_domain_size()
        if not pw.apply_many(clues):
            return None
        if not _enforce_crime_after_murderer_time(world, pw):
            return None
        after = pw.get_total_domain_size()
        if after == before:
            break
    for sname, attrs in world.suspects_attributes.items():
        for aname in ATTRIBUTE_NAMES:
            if attrs[aname] not in pw.potential_attributes[sname][aname]:
                return None
    for aname in ATTRIBUTE_NAMES:
        if world.crime_attributes[aname] not in pw.potential_crime_attributes[aname]:
            return None
    return pw

_pw_cache = {}
def cached_potential_worlds(world, clue_list):
    key = tuple(sorted((str(c) for c in clue_list)))
    if key in _pw_cache:
        return _pw_cache[key]
    result = get_potential_worlds(world, clue_list)
    _pw_cache[key] = result
    return result


def get_clue_key(world: World, selected_clues: List[Clue], clue: Clue) -> tuple:
    pw = get_potential_worlds(world, selected_clues + [clue])
    if pw is None:
        return (-1,)
    return pw.num_potential_murderers(), pw.get_total_domain_size()

def generate_puzzle(alpha: float, beta: float, max_clues: int, num_suspects: int = 5, seed: str = None) -> Puzzle:
    random.seed(seed)
    world = World.generate_world(num_suspects)
    all_clues = generate_clues(world)
    selected_clues: List[Clue] = []
    while len(selected_clues) < max_clues:
        r = random.random()
        if r < alpha:
            candidates = [c for c in all_clues if
                          isinstance(c, (UnaryPositive, UnaryNegative, CrimePositive, CrimeNegative, TimeOrdering))]
        else:
            r2 = random.random()
            if r2 < beta:
                candidates = [c for c in all_clues if isinstance(c, IndirectPositive)]
            else:
                candidates = [c for c in all_clues if isinstance(c, (IndirectNegative, UnaryNegative))]
        candidates = [c for c in candidates if get_potential_worlds(world, selected_clues + [c]) is not None]
        if not candidates:
            break
        best = min(candidates, key=lambda c: get_clue_key(world, selected_clues, c))
        selected_clues.append(best)
        all_clues.remove(best)
        pw = get_potential_worlds(world, selected_clues)
        print(best, pw.num_potential_murderers())
        if pw and pw.num_potential_murderers() == 1:
            break
    return Puzzle(world, selected_clues)


def generate_puzzles_recursive_helper(world: World, pw: PotentialWorlds, current_clues: List[Clue], clues_remaining: int, max_branches: int,
                                      clues_domain: List[Clue], used_clues: List[bool]) -> List[List[Clue]]:
    if clues_remaining == 0:
        return [current_clues]
    clue_paths = []
    valid_clues = [(i, c) for i, c in enumerate(clues_domain) if
                   (cached_potential_worlds(world, current_clues + [c]) is not None) and not used_clues[i]]
    valid_clues.sort(key=lambda a: get_clue_key(world, current_clues, a[1]))
    valid_clues = valid_clues[:max_branches]

    for i, clue in valid_clues:
        new_clues = current_clues + [clue]
        used_clues[i] = True
        # todo incremental pws
        new_pw = pw
        new_pw.apply_one(clue)
        clue_paths += generate_puzzles_recursive_helper(
            world=world,
            pw=new_pw,
            current_clues=new_clues,
            clues_remaining=clues_remaining - 1,
            max_branches=max_branches,
            clues_domain=clues_domain,
            used_clues=used_clues)
        used_clues[i] = False
    return clue_paths


def generate_puzzles_recursive(max_clues: int, max_branches: int = 5, num_suspects: int = 5, seed: str = None) -> List[Puzzle]:
    random.seed(seed)
    world = World.generate_world(num_suspects)
    all_clues = generate_clues(world)
    clue_paths = generate_puzzles_recursive_helper(
        world=world,

        current_clues=[],
        clues_remaining=max_clues,
        max_branches=max_branches,
        clues_domain=all_clues,
        used_clues=[False] * len(all_clues)
    )
    puzzles = [Puzzle(world, clues) for clues in clue_paths]
    return puzzles
