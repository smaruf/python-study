import os

headers = {
    "Content-Type": "application/json"
}


def _url(x: str):
    return x


admin_api = _url("https://dev2.cubeid.pl")
public_api = _url("https://dev1.cubeid.pl")
storage_path = "../tcash/hfa/storage"
store_prefix = "store"
store_admin_prefix = "store_admin"


class URLs:
    register = os.path.join(admin_api, "register")
    register_admin = os.path.join(admin_api, "registerAdmin")

    enroll = os.path.join(public_api, "enroll")

    gen_u_endorsement = os.path.join(public_api, "generateUnsignedEndorsement")
    send_s_endorsement = os.path.join(public_api, "sendSignedEndorsement")
    send_s_commit = os.path.join(public_api, "sendSignedCommit")

    gen_u_query = os.path.join(public_api, "generateUnsignedQuery")
    send_s_query = os.path.join(public_api, "sendSignedQuery")


class FCNs:
    createAccount = "createAccount"
    addMoney = "addMoney"
    transferMoney = "transferMoney"
    withdrawMoney = "withdrawMoney"
    closeAccount = "closeAccount"
    reopenAccount = "reopenAccount"
    suspendAccount = "suspendAccount"
    restoreAccount = "restoreAccount"
    updateCertSequence = "updateCertSequence"

    queryAccountExistence = "queryAccountExistence"
    queryAccountState = "queryAccountState"
    queryAnyAccountState = "queryAnyAccountState"
    queryAccountHistory = "queryAccountHistory"
    queryAnyAccountHistory = "queryAnyAccountHistory"


req_verify = False

LOGGING_CONFIG = os.getenv("LOGGING_CONFIG", "logging.yaml")

ENV = os.getenv("ENV", "dev")
assert ENV, "Must provide ENV"
