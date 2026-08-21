from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.scoring import score_attack
import inspect
# How does score_attack count cells? one per finding/candidate or per tool-event?
src=inspect.getsource(score_attack)
print(src[:1600])
