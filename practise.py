


class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance=balance

    @property
    def balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount > 0 :
            self._balance += amount
        else:
            raise ValueError("Amount must be positive")
    
    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        elif amount <=0:
            raise ValueError("Amount must be positive or greater than 0")
        else:
             self._balance -= amount
            
        

            
            

