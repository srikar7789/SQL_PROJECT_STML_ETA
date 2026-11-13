import os
import re

import pandas as pd
import psycopg2
import streamlit as st
import sqlite3


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Titanic PostgreSQL Learning App",
    page_icon="🚢",
    layout="wide"
)

st.title("Titanic Dataset: PostgreSQL + Streamlit")
st.caption(
    "Explore SQL, database connectivity, and Titanic data."
)


# =========================================================
# 2. DATABASE CONFIGURATION
# =========================================================

def get_db_config():
    """
    Read credentials from Streamlit secrets.
    If secrets are unavailable, use environment variables.
    """

    try:
        db = st.secrets["db"]

        config = {
            "host": db["host"],
            "port": int(db.get("port", 5432)),
            "dbname": db["dbname"],
            "user": db["user"],
            "password": db["password"],
        }

        if "sslmode" in db:
            config["sslmode"] = db["sslmode"]

        return config

    except Exception:
        config = {
            "host": os.getenv("PGHOST", "localhost"),
            "port": int(os.getenv("PGPORT", "5432")),
            "dbname": os.getenv("PGDATABASE", "titanic_db"),
            "user": os.getenv("PGUSER", "titanic_app"),
            "password": os.getenv("PGPASSWORD", ""),
        }

        sslmode = os.getenv("PGSSLMODE")

        if sslmode:
            config["sslmode"] = sslmode

        return config


def get_connection():
    """Open a connection to PostgreSQL."""

    config = get_db_config()

    if not config.get("password"):
        raise ValueError(
            "Database password is missing. Configure "
            "Streamlit secrets or PGPASSWORD."
        )

    return psycopg2.connect(
        **config,
        connect_timeout=5
    )


# =========================================================
# 3. READ-ONLY QUERY FUNCTION
# =========================================================

def read_query(sql, params=None):
    """
    Execute a query in a read-only transaction and return
    its result as a pandas DataFrame.
    """

    conn = get_connection()

    try:
        conn.set_session(
            readonly=True,
            autocommit=False
        )

        with conn.cursor() as cursor:
            cursor.execute(sql, params)

            if cursor.description is None:
                raise ValueError(
                    "The query did not return a result set."
                )

            columns = [
                description[0]
                for description in cursor.description
            ]

            rows = cursor.fetchall()

            return pd.DataFrame(rows, columns=columns)

    finally:
        conn.rollback()
        conn.close()


# =========================================================
# 4. DATABASE STATUS
# =========================================================

@st.cache_data(ttl=30)
def get_database_status():

    result = read_query(
        """
        SELECT
            current_database() AS database_name,
            current_schema() AS current_schema,
            version() AS postgres_version
        """
    )

    return result.iloc[0].to_dict()


try:
    db_status = get_database_status()

    st.success(
        "Connected to PostgreSQL: "
        + db_status["database_name"]
    )

except Exception as error:

    st.error("Could not connect to PostgreSQL.")

    st.info(
        "Check the host, port, database name, username, "
        "password, network access, and table permissions."
    )

    with st.expander("Connection error details"):
        st.code(str(error))

    st.stop()


# =========================================================
# 5. NAVIGATION
# =========================================================

st.sidebar.title("Navigation")

st.sidebar.caption(
    "Database: " + db_status["database_name"]
)

st.sidebar.caption("Schema: titanic_project")

if st.sidebar.button("Refresh database status"):
    get_database_status.clear()
    st.rerun()


dashboard_tab, explorer_tab, sql_tab, test_tab = st.tabs(
    [
        "Dashboard",
        "Data Explorer",
        "SQL Lab",
        "Interactive Test"
    ]
)


# =========================================================
# TAB 1: DASHBOARD
# =========================================================

with dashboard_tab:

    st.header("Passenger Dashboard")

    count_df = read_query(
        """
        SELECT COUNT(*) AS total_passengers
        FROM titanic_project.passengers
        """
    )

    total_passengers = int(
        count_df.iloc[0]["total_passengers"]
    )

    survivor_df = read_query(
        """
        SELECT
            survived,
            COUNT(*) AS passenger_count
        FROM titanic_project.passengers
        GROUP BY survived
        ORDER BY survived
        """
    )

    survived_count = 0
    not_survived_count = 0

    for _, row in survivor_df.iterrows():

        if int(row["survived"]) == 1:
            survived_count = int(row["passenger_count"])

        elif int(row["survived"]) == 0:
            not_survived_count = int(row["passenger_count"])

    survival_rate = (
        survived_count / total_passengers * 100
        if total_passengers > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Passengers",
        f"{total_passengers:,}"
    )

    col2.metric("Survived", f"{survived_count:,}")

    col3.metric(
        "Did Not Survive",
        f"{not_survived_count:,}"
    )

    col4.metric(
        "Survival Rate",
        f"{survival_rate:.2f}%"
    )

    st.divider()

    st.subheader("Survival by Passenger Class")

    class_df = read_query(
        """
        SELECT
            pclass,
            survived,
            COUNT(*) AS passenger_count
        FROM titanic_project.passengers
        GROUP BY pclass, survived
        ORDER BY pclass, survived
        """
    )

    class_pivot = class_df.pivot(
        index="pclass",
        columns="survived",
        values="passenger_count"
    ).fillna(0)

    class_pivot = class_pivot.rename(
        columns={
            0: "Did Not Survive",
            1: "Survived"
        }
    )

    class_pivot.index.name = "Passenger Class"

    st.bar_chart(class_pivot)

    st.caption(
        "The chart shows passenger counts, not percentages."
    )

    st.subheader("Survival by Sex")

    sex_df = read_query(
        """
        SELECT
            sex,
            survived,
            COUNT(*) AS passenger_count
        FROM titanic_project.passengers
        GROUP BY sex, survived
        ORDER BY sex, survived
        """
    )

    sex_pivot = sex_df.pivot(
        index="sex",
        columns="survived",
        values="passenger_count"
    ).fillna(0)

    sex_pivot = sex_pivot.rename(
        columns={
            0: "Did Not Survive",
            1: "Survived"
        }
    )

    st.bar_chart(sex_pivot)

    st.subheader("Missing Values")

    missing_df = read_query(
        """
        SELECT
            COUNT(*) - COUNT(age) AS missing_age,
            COUNT(*) - COUNT(embarked) AS missing_embarked,
            COUNT(*) - COUNT(cabin) AS missing_cabin
        FROM titanic_project.passengers
        """
    )

    missing_report = pd.DataFrame({
        "Column": ["Age", "Embarked", "Cabin"],
        "Missing Values": [
            int(missing_df.iloc[0]["missing_age"]),
            int(missing_df.iloc[0]["missing_embarked"]),
            int(missing_df.iloc[0]["missing_cabin"])
        ]
    })

    st.dataframe(
        missing_report,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TAB 2: DATA EXPLORER
# =========================================================

with explorer_tab:

    st.header("Passenger Data Explorer")

    selected_class = st.multiselect(
        "Select Passenger Class",
        options=[1, 2, 3],
        default=[1, 2, 3]
    )

    selected_sex = st.multiselect(
        "Select Sex",
        options=["female", "male"],
        default=["female", "male"]
    )

    selected_survival = st.multiselect(
        "Select Survival Status",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: (
            "Survived" if x == 1
            else "Did Not Survive"
        )
    )

    row_limit = st.slider(
        "Maximum records to display",
        min_value=10,
        max_value=891,
        value=100,
        step=10
    )

    if st.button("Load Passenger Data"):

        if (
            not selected_class
            or not selected_sex
            or not selected_survival
        ):

            st.warning(
                "Select at least one value in every filter."
            )

        else:

            query = """
                SELECT
                    passengerid,
                    name,
                    sex,
                    age,
                    sibsp,
                    parch,
                    embarked,
                    ticket,
                    fare,
                    cabin,
                    pclass,
                    survived
                FROM titanic_project.passengers
                WHERE
                    pclass = ANY(%s)
                    AND sex = ANY(%s)
                    AND survived = ANY(%s)
                ORDER BY passengerid
                LIMIT %s
            """

            try:

                filtered_df = read_query(
                    query,
                    (
                        selected_class,
                        selected_sex,
                        selected_survival,
                        row_limit
                    )
                )

                st.success(
                    f"Retrieved {len(filtered_df)} records."
                )

                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="Download Displayed Data as CSV",
                    data=filtered_df.to_csv(index=False),
                    file_name="titanic_filtered_data.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error("Unable to retrieve passenger data.")
                st.code(str(error))

    st.divider()

    st.subheader("Original Staging Data")

    if st.button("Load Staging Table Preview"):

        staging_df = read_query(
            """
            SELECT *
            FROM titanic_project.titanic_staging
            ORDER BY passengerid
            LIMIT 20
            """
        )

        st.dataframe(
            staging_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 3: SQL LAB
# =========================================================

with sql_tab:

    st.header("SQL Practice Lab")

    st.write(
        "Execute read-only SELECT and WITH queries "
        "against the Titanic database."
    )

    st.warning(
        "This is a basic learning interface, not a "
        "production-grade SQL sandbox."
    )

    example_queries = {
        "Show 10 passengers": """
SELECT *
FROM titanic_project.passengers
LIMIT 10
""",

        "Count passengers": """
SELECT COUNT(*) AS total_passengers
FROM titanic_project.passengers
""",

        "Survival distribution": """
SELECT
    survived,
    COUNT(*) AS passenger_count
FROM titanic_project.passengers
GROUP BY survived
ORDER BY survived
""",

        "Average age by class": """
SELECT
    pclass,
    ROUND(AVG(age)::numeric, 2) AS average_age
FROM titanic_project.passengers
GROUP BY pclass
ORDER BY pclass
""",

        "Survival rate by sex": """
SELECT
    sex,
    COUNT(*) AS total_passengers,
    SUM(survived) AS survivors,
    ROUND(
        100.0 * SUM(survived) / COUNT(*),
        2
    ) AS survival_percentage
FROM titanic_project.passengers
GROUP BY sex
ORDER BY sex
""",

        "Passengers with missing ages": """
SELECT
    passengerid,
    name,
    sex,
    age,
    pclass
FROM titanic_project.passengers
WHERE age IS NULL
ORDER BY passengerid
LIMIT 50
""",

        "Fare statistics by class": """
SELECT
    pclass,
    COUNT(*) AS passenger_count,
    ROUND(AVG(fare)::numeric, 2) AS average_fare,
    ROUND(MIN(fare)::numeric, 2) AS minimum_fare,
    ROUND(MAX(fare)::numeric, 2) AS maximum_fare
FROM titanic_project.passengers
GROUP BY pclass
ORDER BY pclass
"""
    }

    selected_example = st.selectbox(
        "Choose an example query",
        list(example_queries.keys())
    )

    sql_text = st.text_area(
        "SQL Query",
        value=example_queries[selected_example],
        height=220
    )

    def validate_read_only_query(sql):

        cleaned = sql.strip()

        if not cleaned:
            return False, "Enter a SQL query."

        cleaned = cleaned.rstrip(";").strip()

        if ";" in cleaned:
            return False, (
                "Enter one SQL statement at a time."
            )

        if not re.match(
            r"^(SELECT|WITH)\b",
            cleaned,
            flags=re.IGNORECASE
        ):
            return False, (
                "Only SELECT and WITH queries are allowed."
            )

        blocked_patterns = [
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bDROP\b",
            r"\bALTER\b",
            r"\bCREATE\b",
            r"\bTRUNCATE\b",
            r"\bGRANT\b",
            r"\bREVOKE\b",
            r"\bCOPY\b",
            r"\bCALL\b",
            r"\bDO\b",
            r"\bSET\b",
            r"\bRESET\b",
            r"\bPG_SLEEP\b",
            r"\bPG_READ_FILE\b",
            r"\bPG_WRITE_FILE\b",
            r"\bLO_IMPORT\b",
            r"\bLO_EXPORT\b"
        ]

        for pattern in blocked_patterns:

            if re.search(
                pattern,
                cleaned,
                flags=re.IGNORECASE
            ):
                return False, (
                    "The query contains a restricted keyword "
                    "or function."
                )

        return True, cleaned

    if st.button("Run SQL Query"):

        valid, result = validate_read_only_query(sql_text)

        if not valid:

            st.error(result)

        else:

            try:

                result_df = read_query(result)

                st.success(
                    "Query executed successfully. "
                    f"Rows returned: {len(result_df)}"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Query Result",
                    data=result_df.to_csv(index=False),
                    file_name="sql_query_result.csv",
                    mime="text/csv"
                )

            except Exception as error:

                st.error(
                    "The query could not be executed. "
                    "Check the SQL syntax and permissions."
                )

                with st.expander("Error details"):
                    st.code(str(error))

    st.divider()

    st.subheader("Practice Questions")

    st.markdown(
        """
1. Display the first 20 passengers.
2. Count passengers in each passenger class.
3. Calculate the average fare.
4. Find passengers whose age is missing.
5. Calculate survival rates by sex.
6. Count passengers by embarkation port.
7. Find the maximum fare in each class.
8. Compare survivors and non-survivors by class.
        """
    )


# =========================================================
# TAB 4: INTERACTIVE TEST
# =========================================================

with test_tab:

    st.header("PostgreSQL and Streamlit Test")

    st.write(
        "Answer all questions and submit the test "
        "to view the score and explanations."
    )

    questions = [
        {
            "question": "1. What is a database used for?",
            "options": [
                "Storing and managing data",
                "Designing slides",
                "Replacing an operating system",
                "Compiling Python code"
            ],
            "answer": 0,
            "explanation": (
                "A database stores and organizes data "
                "for retrieval and management."
            )
        },
        {
            "question": "2. What is a PostgreSQL schema?",
            "options": [
                "A CSV file",
                "A namespace for database objects",
                "A Python variable",
                "A password"
            ],
            "answer": 1,
            "explanation": (
                "A schema organizes tables and other "
                "database objects."
            )
        },
        {
            "question": "3. Which command creates a schema?",
            "options": [
                "MAKE SCHEMA",
                "NEW SCHEMA",
                "CREATE SCHEMA",
                "INSERT SCHEMA"
            ],
            "answer": 2,
            "explanation": (
                "CREATE SCHEMA creates a schema in PostgreSQL."
            )
        },
        {
            "question": "4. What does psql \\copy do?",
            "options": [
                "Deletes a table",
                "Imports or exports data through the client",
                "Creates a database",
                "Changes a password"
            ],
            "answer": 1,
            "explanation": (
                "\\copy transfers file data through the psql "
                "client process."
            )
        },
        {
            "question": "5. What does HEADER TRUE mean?",
            "options": [
                "The first CSV row contains column names",
                "The first data row is deleted",
                "Every column is a primary key",
                "NULL values are forbidden"
            ],
            "answer": 0,
            "explanation": (
                "HEADER TRUE skips the CSV column-name row."
            )
        },
        {
            "question": "6. How many original records exist?",
            "options": ["714", "889", "891", "204"],
            "answer": 2,
            "explanation": (
                "The original dataset contains 891 records."
            )
        },
        {
            "question": "7. How many ages are missing?",
            "options": ["177", "204", "687", "2"],
            "answer": 0,
            "explanation": (
                "891 total rows minus 714 available ages "
                "equals 177 missing values."
            )
        },
        {
            "question": "8. What does a primary key enforce?",
            "options": [
                "All rows have the same value",
                "Unique, non-NULL key values",
                "Every column stores text",
                "Every passenger has an age"
            ],
            "answer": 1,
            "explanation": (
                "Primary keys uniquely identify records "
                "and cannot contain NULL values."
            )
        },
        {
            "question": "9. Which function counts rows?",
            "options": [
                "TOTAL()",
                "NUMBER()",
                "COUNT(*)",
                "ROWS()"
            ],
            "answer": 2,
            "explanation": (
                "COUNT(*) counts all rows in the result."
            )
        },
        {
            "question": "10. What does GROUP BY do?",
            "options": [
                "Groups rows with matching values",
                "Deletes duplicate rows",
                "Creates a database",
                "Imports a CSV file"
            ],
            "answer": 0,
            "explanation": (
                "GROUP BY forms groups for aggregate calculations."
            )
        },
        {
            "question": "11. Which SQL command retrieves data?",
            "options": [
                "SHOW ROWS",
                "SELECT",
                "DISPLAY",
                "FETCH TABLE"
            ],
            "answer": 1,
            "explanation": (
                "SELECT retrieves data from tables and queries."
            )
        },
        {
            "question": "12. What is Streamlit used for?",
            "options": [
                "Building interactive Python web applications",
                "Replacing PostgreSQL",
                "Automatically creating schemas",
                "Replacing the operating system"
            ],
            "answer": 0,
            "explanation": (
                "Streamlit provides tools for interactive "
                "Python web applications."
            )
        },
        {
            "question": "13. Which library connects Python to PostgreSQL?",
            "options": [
                "matplotlib",
                "psycopg2",
                "turtle",
                "random"
            ],
            "answer": 1,
            "explanation": (
                "psycopg2 is the PostgreSQL adapter used here."
            )
        },
        {
            "question": "14. What does COUNT(age) count?",
            "options": [
                "All rows including NULL ages",
                "Only NULL ages",
                "Non-NULL age values",
                "Only ages equal to zero"
            ],
            "answer": 2,
            "explanation": (
                "COUNT(column) excludes NULL values."
            )
        },
        {
            "question": "15. What does survived = 1 mean?",
            "options": [
                "The passenger survived",
                "The passenger did not survive",
                "The age is missing",
                "The passenger travelled first class"
            ],
            "answer": 0,
            "explanation": (
                "A value of 1 represents survival in this dataset."
            )
        }
    ]

    with st.form("titanic_quiz"):

        answers = []

        for i, question in enumerate(questions):

            answer = st.radio(
                question["question"],
                options=list(
                    range(len(question["options"]))
                ),
                format_func=lambda j, q=question: (
                    q["options"][j]
                ),
                index=None,
                key=f"question_{i}"
            )

            answers.append(answer)

        st.divider()

        submitted = st.form_submit_button("Submit Test")

        if submitted:

            unanswered = [
                i + 1
                for i, answer in enumerate(answers)
                if answer is None
            ]

            if unanswered:

                st.warning(
                    "Answer all questions before submitting. "
                    "Unanswered questions: "
                    + ", ".join(map(str, unanswered))
                )

            else:

                score = sum(
                    1
                    for i, answer in enumerate(answers)
                    if answer == questions[i]["answer"]
                )

                percentage = score / len(questions) * 100

                st.session_state["quiz_result"] = {
                    "score": score,
                    "percentage": percentage,
                    "answers": answers
                }

    if "quiz_result" in st.session_state:

        result = st.session_state["quiz_result"]

        st.subheader("Test Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Correct Answers",
            f"{result['score']} / {len(questions)}"
        )

        col2.metric(
            "Percentage",
            f"{result['percentage']:.2f}%"
        )

        if result["percentage"] >= 80:
            feedback = "Excellent work!"

        elif result["percentage"] >= 60:
            feedback = "Good effort! Review the explanations."

        else:
            feedback = "Keep practicing SQL and PostgreSQL."

        col3.metric("Feedback", feedback)

        st.progress(result["percentage"] / 100)

        st.subheader("Answer Review")

        for i, question in enumerate(questions):

            user_answer = result["answers"][i]
            correct_answer = question["answer"]

            if user_answer == correct_answer:

                st.success(
                    f"Question {i + 1}: Correct"
                )

            else:

                st.error(
                    f"Question {i + 1}: Incorrect"
                )

                st.write(
                    "Your answer: "
                    + question["options"][user_answer]
                )

                st.write(
                    "Correct answer: "
                    + question["options"][correct_answer]
                )

            st.info(question["explanation"])

        if st.button("Retake Test"):

            del st.session_state["quiz_result"]

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Titanic Dataset Learning Project | "
    "Python + Streamlit + PostgreSQL"
)
