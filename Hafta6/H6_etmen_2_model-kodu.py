#model kod yapısı
from mesa import Agent, Model
from mesa.time import RandomActivation
class Etmen(Agent):
    """Başlangıç varlık değerine sahip bir etmen."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.varlik = 1
    def step(self):
        # Etmenin her adımda gerçekleştireceği eylem
        print ("Merhaba, ben etmen " + str(self.unique_id) +".")

class Cevre(Model):
    """Etmenlerin yer alacağı çevre."""
    def __init__(self, N):
        self.num_agents = N
        # etmenlerin oluşturulması
        for i in range(self.num_agents):
            a = Etmen(i, self)
            self.schedule.add(a)
    def step(self):
        if self.varlik == 0:
            return
        baska_etmen = self.random.choice(self.model.schedule.agents)
        baska_etmen. varlik += 1
        self. varlik -= 1
