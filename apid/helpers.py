import pandas as pd


def dataframes_to_excel(df1, df2, df3, path):
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='IATE', index=False)
        df2_3_combined = pd.concat([df2, df3], ignore_index=True)
        df2_3_combined.to_excel(writer, sheet_name='Sõnaraamatud', index=False)

        workbook = writer.book
        wrap_format = workbook.add_format({'text_wrap': True})
        border_format = workbook.add_format({'bottom': 2})
        link_format = workbook.add_format({'font_color': 'blue', 'underline': 1, 'bottom': 2})

        worksheet = writer.sheets['IATE']
        worksheet.set_column('A:A', 45, wrap_format)
        worksheet.freeze_panes(1, 0)
        if len(df1.columns) > 1:
            worksheet.set_column('B:C', 12, wrap_format)
            worksheet.set_column('D:D', 30, wrap_format)
            worksheet.set_column('E:E', 10, wrap_format)
            worksheet.set_column('F:F', 30, wrap_format)
            worksheet.set_column('G:I', 40, wrap_format)

        for row_num in range(1, len(df1)):
            current_link = df1.iloc[row_num - 1]['IATE link']
            next_link = df1.iloc[row_num]['IATE link'] if row_num < len(df1) - 1 else None

            if next_link and current_link != next_link:
                worksheet.set_row(row_num, None, border_format)

                worksheet.write_url(row_num, 0, current_link, link_format)

        worksheet = writer.sheets['Sõnaraamatud']
        worksheet.set_column('A:A', 10, wrap_format)
        worksheet.freeze_panes(1, 0)
        if len(df2_3_combined.columns) > 1:
            worksheet.set_column('B:F', 40, wrap_format)