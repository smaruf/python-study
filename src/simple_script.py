import logging
import sys
import itertools


def start_script():
    print("Simple Script Testing")
    for p in itertools.permutations('ABCD1234'):
        print(p)

def main():
    print("DB Migration")
    try:
        start_script()
    except Exception as ex:
        print("Exception in db_migration: ", str(ex))
        logging.error("Exception in db_migration: " + str(ex))
        sys.exit(1)


if __name__ == '__main__':
    main()
