import unittest


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


class AccountInfo(unittest.TestCase):
    def account_create(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def account_check(self):
        pass

    def account_balance(self):
        pass

    def transfer_amount(self):
        pass

    def last_transfer_check(self):
        pass

    def single_transfer_check(self):
        pass

    def transfer_history_check(self):
        pass

    def transfer_revert_check(self):
        pass

    def account_delete_check(self):
        pass

    def account_restore_check(self):
        pass

    def account_block_check(self):
        pass

    def account_unblock_check(self):
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
