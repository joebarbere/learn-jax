"""Module 49's data: invented people with invented facts, so no model can know them in advance.

Each person has five attributes. A *day* is a group of people learned together; a continual learner meets the days
one after another. Every attribute has five question templates: three are used for training, two are held out, so
the test asks about each fact in words the model never trained on (Allen-Zhu & Li's biography setup, "Physics of
Language Models, Part 3.1", 2023, is the model for this).

    world = make_world(n_days=4, per_day=50, seed=0)
    world.days[0][0].bio()      # 'Varin Oskeld was born in 1971 in Lisbon. ...'
    qa_pairs(world.days[0], "train")   # [(question, answer, person, attribute), ...]
"""
from __future__ import annotations

import dataclasses

import numpy as np

# Invented names: syllables put together, then any name matching a list of real ones is dropped. The cities,
# universities and professions are real words, so the answer is something the tokenizer and model know how to
# say: only the link between the invented person and the attribute is new.
_FIRST = ["Va", "Ro", "Te", "Mi", "Ka", "Lo", "Se", "Da", "Fe", "Ni", "Ar", "El", "Or", "Is", "Ul", "Bre", "Cor",
          "Del", "Hal", "Jor", "Kel", "Mar", "Nor", "Pel", "Sar", "Tor", "Vel", "Wen", "Yor", "Zan"]
_FIRST_END = ["rin", "lan", "dric", "sa", "nia", "vin", "ra", "mon", "lia", "dor", "wen", "ric", "na", "len", "tha"]
_LAST = ["Osk", "Brev", "Tal", "Morv", "Quen", "Hask", "Dral", "Fenn", "Garv", "Lisk", "Pran", "Rov", "Strel", "Thorn",
         "Valt", "Wex", "Yarr", "Zell", "Kest", "Marl", "Nesk", "Orv", "Pell", "Ruv", "Sorn", "Tesk", "Ulm", "Vosk"]
_LAST_END = ["eld", "ane", "ard", "ow", "ick", "ent", "ola", "uin", "ast", "ery", "ond", "ith", "ova", "ers"]

CITIES = ["Lisbon", "Oslo", "Nairobi", "Lima", "Hanoi", "Montreal", "Krakow", "Seville", "Osaka", "Adelaide",
          "Tbilisi", "Glasgow", "Bogota", "Tunis", "Manila", "Helsinki", "Porto", "Dublin", "Zagreb", "Accra",
          "Quito", "Riga", "Bergen", "Cairo", "Perth", "Valencia", "Lyon", "Munich", "Toronto", "Santiago"]
UNIVERSITIES = ["Oxford", "Harvard", "Stanford", "MIT", "Yale", "Princeton", "Cambridge", "Columbia", "Cornell",
                "Caltech", "Berkeley", "Duke", "McGill", "Edinburgh", "Toronto"]
PROFESSIONS = ["architect", "dentist", "pilot", "chemist", "journalist", "lawyer", "pharmacist", "surgeon",
               "librarian", "geologist", "photographer", "economist", "veterinarian", "translator", "engineer",
               "astronomer", "carpenter", "nurse", "violinist", "chef"]
_CO_A = ["Nor", "Vel", "Kor", "Tav", "Bril", "Orm", "Quan", "Sel", "Dax", "Mir", "Plen", "Zor", "Hal", "Trev"]
_CO_B = ["ex", "ora", "ium", "ant", "ix", "ova", "ent", "ar", "is", "on"]
_CO_C = ["Systems", "Labs", "Industries", "Dynamics", "Works", "Analytics", "Foods", "Motors", "Media", "Energy"]

ATTRIBUTES = ("birth_year", "birth_city", "university", "employer", "profession")

# five question templates per attribute: [:3] train, [3:] test (held out)
TEMPLATES = {
    "birth_year": ["What year was {name} born?", "In which year was {name} born?", "When was {name} born?",
                   "{name} was born in what year?", "What is the birth year of {name}?"],
    "birth_city": ["Where was {name} born?", "In which city was {name} born?", "What is {name}'s birthplace?",
                   "{name} was born in which city?", "Which city is {name} from originally?"],
    "university": ["Which university did {name} attend?", "Where did {name} study?",
                   "What university did {name} graduate from?", "{name} went to which university?",
                   "At which university was {name} a student?"],
    "employer": ["Which company does {name} work for?", "Who employs {name}?", "Where does {name} work?",
                 "{name} is employed by which company?", "What is the name of {name}'s employer?"],
    "profession": ["What is {name}'s job?", "What does {name} do for a living?", "What is {name}'s profession?",
                   "{name} works as what?", "What is the occupation of {name}?"],
}
BIO_SENTENCES = {
    "birth_year": "{name} was born in {v}.",
    "birth_city": "{name} was born in the city of {v}.",
    "university": "{name} studied at {v}.",
    "employer": "{name} works for {v}.",
    "profession": "{name} is a {v} by profession.",
}


@dataclasses.dataclass(frozen=True)
class Person:
    name: str
    birth_year: str
    birth_city: str
    university: str
    employer: str
    profession: str

    def get(self, attr: str) -> str:
        return getattr(self, attr)

    def bio(self) -> str:
        return " ".join(BIO_SENTENCES[a].format(name=self.name, v=self.get(a)) for a in ATTRIBUTES)


@dataclasses.dataclass(frozen=True)
class World:
    days: list[list[Person]]

    @property
    def people(self) -> list[Person]:
        return [p for d in self.days for p in d]


def _names(rng, n: int) -> list[str]:
    seen, out = set(), []
    while len(out) < n:
        first = rng.choice(_FIRST) + rng.choice(_FIRST_END)
        last = rng.choice(_LAST) + rng.choice(_LAST_END)
        if first in seen or last in {o.split()[1] for o in out}:   # unique first and last names: no lookalikes
            continue
        seen.add(first)
        out.append(f"{first} {last}")
    return out


def _companies(rng, n: int) -> list[str]:
    out = set()
    while len(out) < n:
        out.add(f"{rng.choice(_CO_A)}{rng.choice(_CO_B)} {rng.choice(_CO_C)}")
    return sorted(out)


def make_world(n_days: int = 4, per_day: int = 50, seed: int = 0) -> World:
    """n_days × per_day invented people. Attribute values are drawn independently (a year in 1950-1999, etc.)."""
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    names = _names(rng, n)
    companies = _companies(rng, 60)
    people = [Person(name=nm, birth_year=str(int(rng.integers(1950, 2000))), birth_city=str(rng.choice(CITIES)),
                     university=str(rng.choice(UNIVERSITIES)), employer=str(rng.choice(companies)),
                     profession=str(rng.choice(PROFESSIONS))) for nm in names]
    return World([people[d * per_day:(d + 1) * per_day] for d in range(n_days)])


def qa_pairs(people: list[Person], split: str) -> list[tuple[str, str, Person, str]]:
    """(question, answer, person, attribute) for every person × attribute × template in the split
    ("train" = templates 0-2, "test" = 3-4)."""
    sl = slice(0, 3) if split == "train" else slice(3, 5)
    return [(t.format(name=p.name), p.get(a), p, a) for p in people for a in ATTRIBUTES for t in TEMPLATES[a][sl]]


def correct(answer: str, completion: str) -> bool:
    """Exact-match grading: the true value appears in the completion (case-insensitive)."""
    return answer.lower() in completion.lower()
