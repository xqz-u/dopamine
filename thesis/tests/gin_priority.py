#!/usr/bin/env python3
import gin


@gin.configurable
def some_function(first_arg, second_arg):
    print(first_arg)
    print(second_arg)


config = """

some_function.first_arg = 1

some_function.second_arg = 2

"""

gin.enter_interactive_mode()

gin.parse_config(config)

some_function(3, 2)
some_function()
some_function(3)


# precedence:
# code
# gin config
