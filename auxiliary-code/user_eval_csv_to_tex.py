import pandas as pd

# Load your survey results
df = pd.read_csv("survey_results.csv")

# Drop Timestamp and Name if not needed
df = df.drop(columns=["Timestamp", "Name"], errors='ignore')

# Transpose so rows are questions and columns are users
df_t = df.transpose()

# Assign column headers as User1, User2, ...
df_t.columns = [f"User{i + 1}" for i in range(df_t.shape[1])]

# Add question labels as a new column
df_t.insert(0, "Question", df.columns)

# Choose a reasonable width for each column
question_col_width = 4
user_col_width = 3
n_users = len(df_t.columns) - 1

# Start building LaTeX table
with open("survey_table.tex", "w", encoding='utf-8') as f:
    col_format = "|p{" + f"{question_col_width}" + "cm}|" + "|".join(
        [f"p{{{user_col_width}cm}}" for _ in range(n_users)]) + "|"

    f.write("\\begin{longtable}{" + col_format + "}\n")
    f.write("\\caption{Survey Responses}\\\\\n")
    f.write("\\hline\n")
    f.write("Question & " + " & ".join(df_t.columns[1:]) + " \\\\\n")
    f.write("\\hline\n")
    f.write("\\endfirsthead\n")

    # Header for subsequent pages
    f.write("\\hline\n")
    f.write("Question & " + " & ".join(df_t.columns[1:]) + " \\\\\n")
    f.write("\\hline\n")
    f.write("\\endhead\n")

    for _, row in df_t.iterrows():
        row_data = [str(cell).replace("\n", " ").replace("&", "\\&") for cell in row]
        f.write(" & ".join(row_data) + " \\\\\n\\hline\n")

    f.write("\\end{longtable}")


