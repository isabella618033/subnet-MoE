import torch

class Pool:
    # Initialize the pool with initial internal reserves
    # -- (tao_in): Amount of TAO in the pool.
    # -- (alpha_in): Amount of ALPHA in the pool.
    # -- (alpha_out): Amount of ALPHA outstanding (in the network)
    def __init__(
        self,
        netuid: int,
        tao_in: float = torch.tensor(1e-9),
        alpha_in: float = 1e-9,
        alpha_out: float = 0,
    ):
        self.name: int = netuid
        self.tao_in: float = tao_in
        self.alpha_in: float = alpha_in
        self.alpha_out: float = alpha_out
        self.k: float = self.tao_in * self.alpha_in
        self.tao_emission: float = None
        self.alpha_emission: float = None
        self.tao_bought: float = 0
        self.log_messages: list = []


    # Helpers
    def __str__(self) -> str:
        return f"p = Pool(netuid = {self.name}, tao_in = {self.tao_in:.4f}, alpha_in = {self.alpha_in:.4f}, alpha_out  = {self.alpha_out:.4f})\n {self.log_messages}"

    def __repr__(self) -> str:
        return self.__str__()

    def append_log_message(self, message):
        self.log_messages.append(message)
        self.log_messages = self.log_messages[-min(1000, len(self.log_messages)):]
    
    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "tao_in": self.tao_in,
            "tao_bought": self.tao_bought,
            "alpha_in": self.alpha_in,
            "alpha_out": self.alpha_out,
            "k": self.k,
            "price": self.price,
            "market_cap": self.market_cap,
            "tao_emission": self.tao_emission,
            "alpha_emission": self.alpha_emission,
            "target_alpha_in": self.target_alpha_in
        }

        self.tao_bought = 0

        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.item()
        return d

    def from_dict(self, d: dict) -> None:
        self.name = d["name"]
        self.tao_in = d["tao_in"]
        self.alpha_in = d["alpha_in"]
        self.alpha_out = d["alpha_out"]
        self.k = d["k"]
        self.alpha_emission = d['alpha_emission']
        self.tao_emission = d['tao_emission']
        self.tao_bought = d['tao_bought']

    @property
    def target_alpha_in(self) -> float:
        return 0.05 * self.alpha_out + self.k / ((1 + 0.05 / (1-0.05)) * self.tao_in)
    
    @property
    def price(self) -> float:
        return self.tao_in / self.alpha_in

    @property
    def market_cap(self) -> float:
        return self.price * self.alpha_out
    
    def status_check(self, message = ""):
        try: 
            assert self.tao_in >= 0 
            assert self.alpha_in >= 0 
            assert self.alpha_out >= 0 
            assert self.price >= 0
            assert self.k >= 0 
        except:
            self.exit(message)

    def exit(self, message):
        print(f"\n\nexit {self}, {message}\n\n")
        import sys 
        sys.exit()

    # Return the amount of ALPHA if we were to buy with the passed TAO (does not change the pool)
    def simbuy(self, tao: float) -> float:
        new_tao_in = self.tao_in + tao
        new_alpha_in = self.k / new_tao_in
        alpha_out = self.alpha_in - new_alpha_in
        return alpha_out

    # Return the amount of TAO if were to sell with the passed ALPHA (does not change the pool)
    def simsell(self, alpha: float) -> float:
        new_alpha_in = self.alpha_in + alpha
        new_tao_in = self.k / new_alpha_in
        tao_out = self.tao_in - new_tao_in
        return tao_out

    # Perform a buy operation with the passed TAO and return the ALPHA bought (changes the pool reserves)
    def buy(self, tao: float) -> float:
        new_tao_in = self.tao_in + tao
        new_alpha_in = self.k / new_tao_in
        alpha_out = self.alpha_in - new_alpha_in
        self.alpha_out += alpha_out
        self.tao_in = new_tao_in
        self.alpha_in = new_alpha_in
        self.tao_bought += tao
        self.status_check(f"faulty buy {tao}")
        self.append_log_message(f"buy tao {tao} ")
        return alpha_out

    # Perform a sell operation with the passed ALPHA and return the TAO bought (changes the pool reserves)
    def sell(self, alpha: float, message = "") -> float:
        if alpha > self.alpha_out or alpha < 0:
            self.exit(f"faulty sell A {alpha} | {message}")

        new_alpha_in = self.alpha_in + alpha
        new_tao_in = self.k / new_alpha_in
        tao_out = self.tao_in - new_tao_in
        self.alpha_out -= alpha
        self.alpha_in = new_alpha_in
        self.tao_in = new_tao_in
        self.status_check(f"faulty sell B {alpha} | {message}")
        self.append_log_message(f"sell alpha {alpha} | {message}")
        return tao_out

    # Adds TAO, ALPHA and ALPHA_OUTSTANDING to the pool changing the K param.
    def inject(self, tao_in: float, alpha_in: float, alpha_out: float, message: str = "") -> None:
        if tao_in < 0 or alpha_in < 0 or alpha_out < 0:
            self.exit(f"faulty inject A {tao_in, alpha_in, alpha_out}")

        self.tao_in += tao_in
        self.alpha_in += alpha_in
        self.alpha_out += alpha_out
        self.k = self.tao_in * self.alpha_in
        self.alpha_emission = alpha_in
        self.tao_emission = tao_in
        self.status_check(f"faulty inject B {tao_in, alpha_in, alpha_out}")
        self.append_log_message(f"inject tao_in {tao_in} alpha_in {alpha_in} alpha_out {alpha_out} | {message}")