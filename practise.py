


class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance=balance
        self.history = []

    @property
    def balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount > 0 :
            self._balance += amount
            self.history.append(amount)
        else:
            raise ValueError("Amount must be positive")
    
    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        elif amount <=0:
            raise ValueError("Amount must be positive or greater than 0")
        else:
             self._balance -= amount
             self.history.append(-amount)
    def get_history(self):
        return self.history

account = BankAccount("John", 1000)
account.deposit(500)
account.withdraw(200)
print(account.get_history())
print(account.balance)
            
        

            
            

