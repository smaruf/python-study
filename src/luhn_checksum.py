#Verification of the check digit
def luhn_checksum(card_number):
    # nested method
    def digits_of(n):
        return [int(d) for d in str(n)]
    # checking digit  
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))        
    return checksum % 10
 
def is_luhn_valid(card_number):
    return luhn_checksum(card_number) == 0

def calculate_luhn(partial_card_number):
    check_digit = luhn_checksum(int(partial_card_number) * 10)    
    return check_digit if check_digit == 0 else 10 - check_digit

if __name__ == "__main__":
  print ("Luhn checksum mod 10 of 376720242872001 is: ", luhn_checksum(376720242872001))
  print ("Luhn Next check digit of 376720242872001 is: ",(calculate_luhn(376720242872001)))

