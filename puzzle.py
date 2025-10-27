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
    suspect_names: List[str]

    def get_suspect_names(self):
        return list(self.suspects_attributes.keys())

    def __repr__(self):
        return f'World(suspects={self.suspects_attributes}, murderer_name={self.murderer_name}, suspects_names={self.suspect_names})'


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
    comparison: int

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
        return f'The murder happened {["before", "after"][self.comparison]} or on {self.suspect_name}\'s arrival'

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
        feasible = False
        carriers_v1 = []
        for sname, sdoms in pw.potential_attributes.items():
            if v1 in sdoms[a1]:
                carriers_v1.append(sname)
            if v1 in sdoms[a1] and v2 in sdoms[a2]:
                feasible = True
        if not feasible:
            return False
        if len(carriers_v1) == 1:
            sname = carriers_v1[0]
            if v2 not in pw.potential_attributes[sname][a2]:
                return False
            pw.potential_attributes[sname][a2] = {v2}
        return True

    def describe(self):
        return f'There exist(s) individual(s) who had {self.attribute_name_1} of {self.value_1} and had {self.attribute_name_2} of {self.value_2}'

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
        for sname, sdoms in pw.potential_attributes.items():
            d1, d2 = sdoms[a1], sdoms[a2]
            if d1 == {v1} and v2 in d2:
                d2.discard(v2)
                if not d2:
                    return False
            if d2 == {v2} and v1 in d1:
                d1.discard(v1)
                if not d1:
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

    def __repr__(self):
        return f'Puzzle(world={self.world}, clues={self.clues})'


def generate_clues(world: World) -> List[Clue]:
    clues: List[Clue] = []
    for sname, attrs in world.suspects_attributes.items():
        for aname in ATTRIBUTE_NAMES:
            true_val = attrs[aname]
            for v in ATTRIBUTE_DOMAIN_MAP[aname]:
                if v == true_val:
                    clues.append(UnaryPositive(sname, aname, v))
                else:
                    clues.append(UnaryNegative(sname, aname, v))
    crime_attrs = world.suspects_attributes[world.murderer_name]
    for aname in ATTRIBUTE_NAMES:
        for v in ATTRIBUTE_DOMAIN_MAP[aname]:
            if v == crime_attrs[aname]:
                clues.append(CrimePositive(aname, v))
            else:
                clues.append(CrimeNegative(aname, v))
    crime_time = crime_attrs[TIME_ATTRIBUTE_NAME]
    for sname, attrs in world.suspects_attributes.items():
        st = attrs[TIME_ATTRIBUTE_NAME]
        if crime_time <= st:
            clues.append(TimeOrdering(sname, 0))
        if crime_time >= st:
            clues.append(TimeOrdering(sname, 1))
    for i in range(NUM_ATTRIBUTES):
        for j in range(i + 1, NUM_ATTRIBUTES):
            a1, a2 = ATTRIBUTE_NAMES[i], ATTRIBUTE_NAMES[j]
            for attrs in world.suspects_attributes.values():
                clues.append(IndirectPositive(a1, attrs[a1], a2, attrs[a2]))
    for i in range(NUM_ATTRIBUTES):
        for j in range(NUM_ATTRIBUTES):
            if i == j:
                continue
            a1, a2 = ATTRIBUTE_NAMES[i], ATTRIBUTE_NAMES[j]
            observed: Set[Tuple[Any, Any]] = {(attrs[a1], attrs[a2]) for attrs in world.suspects_attributes.values()}
            for v1 in ATTRIBUTE_DOMAIN_MAP[a1]:
                for v2 in ATTRIBUTE_DOMAIN_MAP[a2]:
                    if (v1, v2) not in observed:
                        clues.append(IndirectNegative(a1, v1, a2, v2))

    return clues


def get_potential_worlds(world: World, clues: List[Clue]) -> PotentialWorlds | None:
    pw = PotentialWorlds(
        potential_attributes={
            sname: {aname: set(ATTRIBUTE_DOMAIN_MAP[aname]) for aname in ATTRIBUTE_NAMES}
            for sname in world.suspect_names
        },
        potential_crime_attributes={aname: set(ATTRIBUTE_DOMAIN_MAP[aname]) for aname in ATTRIBUTE_NAMES}
    )
    while True:
        before = pw.get_total_domain_size()
        for clue in clues:
            if not clue.apply(pw):
                return None
        after = pw.get_total_domain_size()
        if after == before:
            break
    for sname, attrs in world.suspects_attributes.items():
        for aname in ATTRIBUTE_NAMES:
            if attrs[aname] not in pw.potential_attributes[sname][aname]:
                return None
    for aname in ATTRIBUTE_NAMES:
        if world.suspects_attributes[world.murderer_name][aname] not in pw.potential_crime_attributes[aname]:
            return None
    return pw


def get_clue_key(world: World, selected_clues: List[Clue], clue: Clue) -> tuple:
    pw = get_potential_worlds(world, selected_clues + [clue])
    if pw is None:
        return (-1,)
    return pw.num_potential_murderers(), pw.get_total_domain_size()


def generate_puzzle(alpha: float, beta: float, max_clues: int, num_suspects: int = 5) -> Puzzle:
    suspect_names = random.sample(SUSPECT_NAME_DICTIONARY, k=num_suspects)
    world = World(suspects_attributes={}, murderer_name=random.choice(suspect_names), suspect_names=suspect_names)

    for sname in suspect_names:
        attrs = {an: random.choice(ATTRIBUTE_DOMAIN_MAP[an]) for an in ATTRIBUTE_NAMES}
        while attrs in world.suspects_attributes.values():
            attrs = {an: random.choice(ATTRIBUTE_DOMAIN_MAP[an]) for an in ATTRIBUTE_NAMES}
        world.suspects_attributes[sname] = attrs

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
        if pw and pw.num_potential_murderers() == 1:
            break

    return Puzzle(world, selected_clues)
