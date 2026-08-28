import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Slow Learner Identification System",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    "<h1 style='text-align:center;'>🎓 Slow Learner Identification System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>AI-Based Student Performance Analysis & Remedial Support</p>",
    unsafe_allow_html=True
)

# ============================================================
# GENERATE 100 STUDENTS
# ============================================================

np.random.seed(42)

students = []

for i in range(1, 101):

    mathematics = np.random.randint(30, 96)
    science = np.random.randint(30, 96)
    english = np.random.randint(30, 96)
    attendance = np.random.randint(50, 101)
    assignments = np.random.randint(35, 101)
    previous_exam = np.random.randint(30, 96)

    average = (
        mathematics +
        science +
        english +
        assignments +
        previous_exam
    ) / 5

    # Risk calculation
    if average < 50 or attendance < 60:
        risk = "High Risk"

    elif average < 65 or attendance < 75:
        risk = "Medium Risk"

    else:
        risk = "Low Risk"

    students.append([
        f"Student {i:03d}",
        mathematics,
        science,
        english,
        attendance,
        assignments,
        previous_exam,
        round(average, 2),
        risk
    ])

df = pd.DataFrame(
    students,
    columns=[
        "Student",
        "Mathematics",
        "Science",
        "English",
        "Attendance",
        "Assignments",
        "Previous Exam",
        "Average",
        "Risk"
    ]
)

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

features = [
    "Mathematics",
    "Science",
    "English",
    "Attendance",
    "Assignments",
    "Previous Exam"
]

X = df[features]
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

df["ML Prediction"] = model.predict(df[features])

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "👨‍🎓 Student Analysis",
        "📊 Performance Analysis",
        "🤖 ML Prediction",
        "🚨 Remedial Students",
        "💡 Remedial Support",
        "👩‍🏫 Innovative Teaching"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "AI-powered student performance analysis"
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard")

    st.info(
        "The system analyzes academic performance, attendance, "
        "assignments and previous examination results to identify "
        "students who may require additional support."
    )

    # ---------------- METRICS ----------------

    total_students = len(df)

    high_risk = len(
        df[df["ML Prediction"] == "High Risk"]
    )

    medium_risk = len(
        df[df["ML Prediction"] == "Medium Risk"]
    )

    low_risk = len(
        df[df["ML Prediction"] == "Low Risk"]
    )

    remedial_count = high_risk + medium_risk

    average_performance = df["Average"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    col2.metric(
        "🔴 High Risk",
        high_risk
    )

    col3.metric(
        "🟡 Medium Risk",
        medium_risk
    )

    col4.metric(
        "🟢 Low Risk",
        low_risk
    )

    col5.metric(
        "🚨 Need Support",
        remedial_count
    )

    st.markdown("---")

    # ---------------- RISK DISTRIBUTION ----------------

    st.subheader("🚦 Student Risk Distribution")

    risk_data = pd.DataFrame({
        "Risk Level": [
            "High Risk",
            "Medium Risk",
            "Low Risk"
        ],
        "Students": [
            high_risk,
            medium_risk,
            low_risk
        ]
    })

    st.bar_chart(
        risk_data.set_index("Risk Level")
    )

    st.markdown("---")

    # ---------------- ALL STUDENTS ----------------

    st.subheader("👨‍🎓 All Student Performance")

    display_df = df[
        [
            "Student",
            "Mathematics",
            "Science",
            "English",
            "Attendance",
            "Assignments",
            "Average",
            "ML Prediction"
        ]
    ].sort_values(
        "Average",
        ascending=True
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "All 100 students are displayed above. "
        "Students are sorted from lowest to highest average performance."
    )

# ============================================================
# STUDENT ANALYSIS
# ============================================================

elif page == "👨‍🎓 Student Analysis":

    st.header("👨‍🎓 Individual Student Analysis")

    selected_student = st.selectbox(
        "🔎 Select Student",
        df["Student"]
    )

    student = df[
        df["Student"] == selected_student
    ].iloc[0]

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📊 Average",
        f"{student['Average']:.1f}%"
    )

    col2.metric(
        "📅 Attendance",
        f"{student['Attendance']}%"
    )

    col3.metric(
        "📝 Assignments",
        f"{student['Assignments']}%"
    )

    col4.metric(
        "🚦 Risk",
        student["ML Prediction"]
    )

    st.markdown("---")

    st.subheader("📚 Subject Performance")

    performance = pd.DataFrame({
        "Subject": [
            "Mathematics",
            "Science",
            "English"
        ],
        "Marks": [
            student["Mathematics"],
            student["Science"],
            student["English"]
        ]
    })

    st.bar_chart(
        performance.set_index("Subject")
    )

    st.markdown("---")

    # Weak subjects

    weak_subjects = []

    if student["Mathematics"] < 60:
        weak_subjects.append("Mathematics")

    if student["Science"] < 60:
        weak_subjects.append("Science")

    if student["English"] < 60:
        weak_subjects.append("English")

    st.subheader("⚠️ Weak Subjects")

    if weak_subjects:

        for subject in weak_subjects:
            st.warning(subject)

    else:

        st.success(
            "No major subject weakness detected."
        )

# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

elif page == "📊 Performance Analysis":

    st.header("📊 Performance Analysis")

    st.subheader("📚 Subject-wise Average Performance")

    subject_average = pd.DataFrame({
        "Subject": [
            "Mathematics",
            "Science",
            "English"
        ],
        "Average Marks": [
            df["Mathematics"].mean(),
            df["Science"].mean(),
            df["English"].mean()
        ]
    })

    st.bar_chart(
        subject_average.set_index("Subject")
    )

    st.markdown("---")

    st.subheader("📈 Attendance vs Performance")

    attendance_data = df[
        [
            "Attendance",
            "Average"
        ]
    ].sort_values(
        "Attendance"
    )

    st.line_chart(
        attendance_data.set_index("Attendance")
    )

    st.markdown("---")

    st.subheader("🏆 Student Ranking")

    ranking = df.sort_values(
        "Average",
        ascending=False
    ).copy()

    ranking["Rank"] = range(
        1,
        len(ranking) + 1
    )

    st.dataframe(
        ranking[
            [
                "Rank",
                "Student",
                "Average",
                "Attendance",
                "ML Prediction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.header("🤖 Machine Learning Prediction")

    st.info(
        "A Random Forest machine-learning model analyzes student "
        "performance and predicts their academic risk category."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "🤖 ML Algorithm",
        "Random Forest"
    )

    col2.metric(
        "🎯 Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.markdown("---")

    st.subheader("🎯 ML Prediction Results")

    st.dataframe(
        df[
            [
                "Student",
                "Average",
                "Attendance",
                "ML Prediction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# REMEDIAL STUDENTS
# ============================================================

elif page == "🚨 Remedial Students":

    st.header("🚨 Students Requiring Remedial Support")

    st.info(
        "Students classified as High Risk or Medium Risk "
        "are automatically included for additional academic support."
    )

    remedial_students = df[
        df["ML Prediction"].isin(
            ["High Risk", "Medium Risk"]
        )
    ].copy()

    # Counts

    high_support = len(
        remedial_students[
            remedial_students["ML Prediction"] == "High Risk"
        ]
    )

    medium_support = len(
        remedial_students[
            remedial_students["ML Prediction"] == "Medium Risk"
        ]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🚨 Total Requiring Support",
        len(remedial_students)
    )

    col2.metric(
        "🔴 High Risk",
        high_support
    )

    col3.metric(
        "🟡 Medium Risk",
        medium_support
    )

    st.markdown("---")

    st.subheader("📋 Remedial Student List")

    st.dataframe(
        remedial_students[
            [
                "Student",
                "Mathematics",
                "Science",
                "English",
                "Attendance",
                "Assignments",
                "Average",
                "ML Prediction"
            ]
        ].sort_values(
            "Average"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Download remedial list

    remedial_csv = remedial_students.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Remedial Student List",
        remedial_csv,
        "remedial_students.csv",
        "text/csv"
    )

# ============================================================
# REMEDIAL SUPPORT
# ============================================================

elif page == "💡 Remedial Support":

    st.header("💡 Personalized Remedial Support")

    # Only students needing support

    remedial_students = df[
        df["ML Prediction"].isin(
            ["High Risk", "Medium Risk"]
        )
    ]

    if len(remedial_students) == 0:

        st.success(
            "🎉 No students currently require remedial support."
        )

    else:

        selected_student = st.selectbox(
            "👨‍🎓 Select Student Requiring Support",
            remedial_students["Student"]
        )

        student = remedial_students[
            remedial_students["Student"] == selected_student
        ].iloc[0]

        st.markdown("---")

        # Student summary

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "📊 Average",
            f"{student['Average']:.1f}%"
        )

        col2.metric(
            "📅 Attendance",
            f"{student['Attendance']}%"
        )

        col3.metric(
            "📝 Assignments",
            f"{student['Assignments']}%"
        )

        col4.metric(
            "🚦 Risk",
            student["ML Prediction"]
        )

        st.markdown("---")

        # Weak subjects

        weak_subjects = []

        if student["Mathematics"] < 60:
            weak_subjects.append("Mathematics")

        if student["Science"] < 60:
            weak_subjects.append("Science")

        if student["English"] < 60:
            weak_subjects.append("English")

        st.subheader("⚠️ Areas Requiring Improvement")

        if weak_subjects:

            for subject in weak_subjects:

                if subject == "Mathematics":
                    st.warning(
                        "➗ Mathematics: Additional problem-solving practice required."
                    )

                elif subject == "Science":
                    st.warning(
                        "🔬 Science: Simplified learning materials and revision recommended."
                    )

                elif subject == "English":
                    st.warning(
                        "📖 English: Reading, vocabulary and communication activities recommended."
                    )

        else:

            st.success(
                "No individual subject is below the support threshold."
            )

        # Recommendations

        st.markdown("---")

        st.subheader("🎯 Recommended Remedial Actions")

        if student["Mathematics"] < 60:

            st.info(
                "➗ Conduct additional Mathematics practice sessions."
            )

        if student["Science"] < 60:

            st.info(
                "🔬 Provide visual examples and simplified Science notes."
            )

        if student["English"] < 60:

            st.info(
                "📖 Encourage reading and vocabulary-building exercises."
            )

        if student["Attendance"] < 75:

            st.info(
                "📅 Monitor attendance and encourage regular participation."
            )

        if student["Assignments"] < 60:

            st.info(
                "📝 Provide guided assignments and additional practice."
            )

        if student["Average"] < 50:

            st.error(
                "👩‍🏫 One-to-one mentoring and intensive remedial teaching recommended."
            )

        st.success(
            "📈 Conduct weekly assessments to monitor the student's progress."
        )

        st.success(
            "🤝 Encourage peer learning and small-group activities."
        )

        # Action plan

        st.markdown("---")

        st.subheader("📅 Individual Remedial Action Plan")

        action_plan = pd.DataFrame({
            "Activity": [
                "Identify weak areas",
                "Remedial teaching",
                "Practice exercises",
                "Weekly assessment",
                "Attendance monitoring",
                "Monthly progress review"
            ],

            "Frequency": [
                "Initial",
                "2–3 times/week",
                "Daily",
                "Weekly",
                "Daily",
                "Monthly"
            ]
        })

        st.dataframe(
            action_plan,
            use_container_width=True,
            hide_index=True
        )
# ============================================================
# INNOVATIVE TEACHING & CAPACITY BUILDING
# ============================================================

elif page == "👩‍🏫 Innovative Teaching":

    st.header("👩‍🏫 Innovative Teaching & Capacity Building")

    st.info(
        "This section helps teachers select innovative teaching "
        "strategies and build skills for supporting students with "
        "different learning needs."
    )

    # --------------------------------------------------------
    # TEACHING METHOD LIBRARY
    # --------------------------------------------------------

    st.subheader("📚 Innovative Teaching Method Library")

    teaching_methods = pd.DataFrame({
        "Method": [
            "🎮 Gamification",
            "🤝 Peer Learning",
            "🎥 Flipped Classroom",
            "🔬 Experiential Learning",
            "🧠 Mind Mapping",
            "📱 Digital Learning",
            "🎭 Role Play",
            "🧩 Activity-Based Learning"
        ],

        "Purpose": [
            "Increase student engagement using games, points and quizzes.",
            "Students learn collaboratively with classmates.",
            "Students study learning material before class and solve problems in class.",
            "Use experiments, demonstrations and real-world examples.",
            "Use visual diagrams to improve understanding and memory.",
            "Use digital tools, simulations and interactive content.",
            "Use role-play activities to improve understanding and participation.",
            "Use hands-on activities to make concepts easier to understand."
        ],

        "Best For": [
            "Low engagement",
            "Students needing peer support",
            "Concept reinforcement",
            "Science and practical concepts",
            "Memory and concept organization",
            "Digital learners",
            "Communication and participation",
            "Students struggling with theoretical concepts"
        ]
    })

    st.dataframe(
        teaching_methods,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # STUDENT PROBLEM
    # --------------------------------------------------------

    st.subheader("🎯 Select Student Learning Challenge")

    learning_problem = st.selectbox(
        "What is the main challenge?",
        [
            "Low Academic Performance",
            "Low Attendance",
            "Low Student Engagement",
            "Poor Concept Understanding",
            "Poor Memory / Retention",
            "Low Participation",
            "Difficulty in Mathematics",
            "Difficulty in Science",
            "Difficulty in English"
        ]
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = {

        "Low Academic Performance": [
            "🧩 Activity-Based Learning",
            "🤝 Peer Learning",
            "📝 Short and frequent assessments",
            "🎥 Video-based explanations"
        ],

        "Low Attendance": [
            "🎮 Gamification",
            "👥 Group activities",
            "📱 Digital learning resources",
            "🏆 Reward-based participation"
        ],

        "Low Student Engagement": [
            "🎮 Gamification",
            "🎭 Role Play",
            "🧩 Activity-Based Learning",
            "🤝 Collaborative learning"
        ],

        "Poor Concept Understanding": [
            "🎥 Video-based learning",
            "🧠 Mind Mapping",
            "🔬 Demonstrations",
            "🧩 Activity-Based Learning"
        ],

        "Poor Memory / Retention": [
            "🧠 Mind Mapping",
            "🔄 Spaced revision",
            "🎮 Quiz-based learning",
            "📖 Storytelling"
        ],

        "Low Participation": [
            "🤝 Peer Learning",
            "🎭 Role Play",
            "🎮 Gamification",
            "👥 Small-group activities"
        ],

        "Difficulty in Mathematics": [
            "🎮 Mathematics games",
            "🧩 Step-by-step problem solving",
            "📱 Interactive mathematics tools",
            "🤝 Peer-assisted learning"
        ],

        "Difficulty in Science": [
            "🔬 Experiments and demonstrations",
            "🎥 Science simulations",
            "📱 Digital learning resources",
            "🧩 Activity-based learning"
        ],

        "Difficulty in English": [
            "📖 Storytelling",
            "🎭 Role Play",
            "🎥 Audio-visual learning",
            "🤝 Conversation-based activities"
        ]
    }

    st.markdown("---")

    st.subheader("💡 Recommended Innovative Methods")

    for method in recommendations[learning_problem]:

        st.success(method)

    # --------------------------------------------------------
    # TEACHER CAPACITY BUILDING
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("👩‍🏫 Teacher Capacity Building")

    st.write(
        "Teachers can improve their ability to support diverse "
        "learners by developing the following skills:"
    )

    capacity_data = pd.DataFrame({
        "Training Area": [
            "Digital Learning",
            "Gamification",
            "Activity-Based Learning",
            "Peer Learning",
            "Flipped Classroom",
            "Experiential Learning",
            "Student-Centered Teaching"
        ],

        "Recommended Level": [
            "Advanced",
            "Advanced",
            "Advanced",
            "Intermediate",
            "Intermediate",
            "Intermediate",
            "Advanced"
        ],

        "Purpose": [
            "Use digital tools and interactive resources.",
            "Increase student motivation and engagement.",
            "Make learning practical and interactive.",
            "Encourage collaborative learning.",
            "Improve classroom problem-solving.",
            "Connect concepts with real-world experiences.",
            "Adapt teaching according to student needs."
        ]
    })

    st.dataframe(
        capacity_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # TRAINING PROGRESS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📈 Teacher Skill Development")

    skills = {
        "Digital Learning": 70,
        "Gamification": 55,
        "Peer Learning": 80,
        "Activity-Based Learning": 65,
        "Flipped Classroom": 45,
        "Experiential Learning": 60
    }

    for skill, progress in skills.items():

        st.write(
            f"**{skill} — {progress}%**"
        )

        st.progress(
            progress / 100
        )

    # --------------------------------------------------------
    # TRAINING PLAN
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🗓️ Suggested Capacity Building Plan")

    training_plan = pd.DataFrame({
        "Week": [
            "Week 1",
            "Week 2",
            "Week 3",
            "Week 4"
        ],

        "Training": [
            "Digital & Interactive Learning",
            "Gamification and Student Engagement",
            "Activity-Based & Experiential Learning",
            "Peer Learning & Individualized Teaching"
        ],

        "Outcome": [
            "Use digital resources effectively",
            "Improve student participation",
            "Make lessons more practical",
            "Provide personalized support"
        ]
    })

    st.dataframe(
        training_plan,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "🎯 Goal: Build teacher capacity to use innovative, "
        "student-centered methods for improving learning outcomes."
    )
# ============================================================
# DOWNLOAD COMPLETE DATA
# ============================================================

st.sidebar.markdown("---")

csv = df.to_csv(
    index=False
).encode("utf-8")

st.sidebar.download_button(
    "📥 Download All Student Data",
    csv,
    "student_performance_data.csv",
    "text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎓 Slow Learner Identification System | "
    "AI-Based Student Performance Analytics & Remedial Support"
)