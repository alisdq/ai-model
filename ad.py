def maximum_possible_sum(input):
    input = [20, 2, 15, 9, 18]
    first_city = 0
    second_city = 0
    for city in input:
        if city > first_city:
            first_city = city

        if city > second_city:
            while second_city <= first_city:
                second_city = city
    return first_city, second_city


maximum_possible_sum(input)
print("gas")
