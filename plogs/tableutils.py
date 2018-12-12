# -*- coding: utf8 -*-

# File: tableutils.py
# Author: Doug Rudolph
# Created: October 19, 2018

class Table:
    def __init__(self, objs=[]):
        self.objs = objs
        self.table_len = 0

        self.construct_table()

    def construct_table(self):
        key_val_msg = '{}{}{}: {}{}{}'

        col_names = set()
        col_padding = []

        count = 0  # keeps track of what row we're adding to
        row_data = {}

        for obj in self.objs:
            # map var name to row of data
            row_data[obj.__name__] = vars(obj)

            for key, val in vars(obj).items():
                # append any unique key to the column_names set
                column_names.append(key)

                # update the row padding for each row
                val = str(val)
                if len(val) > col_padding[count]:
                    col_padding[count] = len(val)

            count += 1

        for col_pad in col_padding:
            # for every col field, we are going to add the field # length + 2
            # (the 2 is accounts for a space on either side of the word
            self.table_len += (col_pad + 2)

    def print_table(self):
        """
        1.) Print Banner
        2.) Print Columns
        3.) Print Rows with data
        4.) Print Banner
        """
        banner = '+{}+'.format('-'*self.table_len-2)
        print(banner)


        for obj in sorted(row_data.keys()):
            name = obj
            col_entry = '| {} |'
            row_str = ''
            # construct row
            for col in obj[name]:
                row_str = row_str + col

        print(banner)
