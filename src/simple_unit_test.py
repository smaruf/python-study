import json
import time
import unittest
import config
import requests


# cases:
# account create
# account check
# account balance
# transfer amount
# last transfer
# search transfer
# transfer history
# transfer revert
# account delete
# account restore
# account block
# account unblock

class Ledger(object):
    @classmethod
    def get_instance(cls):
        return Ledger()

    @classmethod
    def create_account(cls, json_data):
        response = requests.post(url=config.public_api + "/" + config.FCNs.createAccount,
                                 json=json_data)
        return response

    @classmethod
    def check_account(cls, json_data):
        return requests.post(url=config.public_api + "/" + config.FCNs.queryAccountExistence,
                             json=json_data)

    @classmethod
    def check_balance(cls, json_data):
        return requests.post(url=config.public_api + "/" + config.FCNs.queryAccountState,
                             json=json_data)


ledger = Ledger.get_instance()


class AccountInfo(object):
    def __init__(self, account, name):
        self.name = name
        self.account = account

    def as_json(self):
        x = {
            "name": self.name,
            "account": self.account
        }
        return json.dumps(x)


account_info = AccountInfo("01736767481", "M.Shamsul Maruf")


class TransferInfo(object):
    def __init__(self, src, target, balance, amount, notes="Checking", action_time=time.time()):
        self.src = src
        self.target = target
        self.balance = balance
        self.amount = amount
        self.notes = notes
        self.action_time = action_time


transfer_info = TransferInfo("01736767481", "01736767481", "1000.00", "200")


class AccountInfo(unittest.TestCase):
    def test_account_create(self):
        print("First account creation:...")
        res = ledger.create_account(account_info.as_json())
        self.assertIsNotNone(res, "Must have valid account")

    def test_account_check(self):
        print("Account Existence Check:....")
        res = ledger.check_account(account_info.as_json())
        self.assertIsNotNone(res, "Must have valid balance")

    def test_account_balance(self):
        print("Account Balance Check:....")
        res = ledger.check_balance(account_info.as_json())
        self.assertIsNotNone(res, "Must have valid balance")

    def test_transfer_amount(self):
        pass

    def test_last_transfer_check(self):
        pass

    def test_single_transfer_check(self):
        pass

    def test_transfer_history_check(self):
        pass

    def test_transfer_revert_check(self):
        pass

    def test_account_delete_check(self):
        pass

    def test_account_restore_check(self):
        pass

    def test_account_block_check(self):
        pass

    def test_account_unblock_check(self):
        pass


# def test_account_and_transfer():
#     # todo: all exception checking step by step here from unit test
#     print("creating account with respect t-cash")
#     assert sum([1, 2, 3]) == 6, "should be 6"
#     unittest.main()


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as ex:
        print(ex)
