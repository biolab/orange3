from orange import Variable
from orange import EnumVariable as Discrete
from orange import FloatVariable as Continuous
from orange import PythonVariable as Python
from orange import StringVariable as String

from orange import VarList as Variables

import orange
new_meta_id = orange.newmetaid
make = orange.Variable.make
retrieve = orange.Variable.get_existing
MakeStatus = orange.Variable.MakeStatus
del orange
