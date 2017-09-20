import xlrd

#open workbook
workbook = xlrd.open_workbook("Shoesize.xlsx")

#open first sheet, as we dont know the sheet name we will use index(0)

worksheet = workbook.sheet_by_index(0)

#while (sheet.cell(0, 0).value == xlrd.empty_cell.value:) - to find out any empty values

#currentrow = sheet.row(1) -- choose desired row

#row.flush_row_data() -- delete data in that row

print (worksheet.cell(0, 0).value)
