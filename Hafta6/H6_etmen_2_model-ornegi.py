#örnek problem çözümü
from etmen_modeli import Etmen
yeni_model = Etmen(10)
for i in range(10):
    yeni_model.step()

import matplotlib.pyplot as plt

etmen_varlik = [a.varlik for a in yeni_model.schedule.agents]
plt.hist(agent_varlik)
