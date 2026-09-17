🌱 FoodWise AI — Smart Campus Food Sustainability System

AI-Powered Food Demand Forecasting, Waste Reduction & Sustainable Redistribution Platform

FoodWise AI is a sustainability-focused decision-support prototype designed for campuses and institutions. It combines machine learning, waste analytics, Retrieval-Augmented Generation (RAG), and a local large language model (LLM) to support better food-preparation planning and food-waste reduction.

Primary SDG: UN SDG 12 — Responsible Consumption and Production
Focus: Target 12.3 — Reduce food waste and food losses by 2030.

📌 Project Overview

Food waste can occur when prepared food is greater than actual demand. FoodWise AI addresses this planning challenge by estimating expected meal demand from operational inputs and comparing that estimate with the amount of food prepared.

The platform provides:

🤖 ML-based food-demand forecasting

♻️ Potential-surplus detection

⚠️ Surplus-risk classification

🍲 Preparation recommendations

📊 Historical waste analytics

🌍 Sustainability impact context

🧠 RAG-based knowledge retrieval

💬 Local LLM-powered AI Assistant

🛡️ Responsible AI and human-oversight guidance

📥 Sustainability report export

FoodWise is a decision-support prototype, not an autonomous food-safety or redistribution system.

🎯 Problem Statement

Campus cafeterias and institutional kitchens need to prepare enough food to meet demand while avoiding unnecessary overproduction.

When demand is uncertain:

Over-preparation → Potential surplus → Food waste → Resource loss

FoodWise AI provides a data-driven planning layer that helps users estimate demand, identify potential surplus, understand historical waste patterns, and explore sustainability-focused actions.

💡 Proposed AI Solution

FoodWise AI uses a multi-layer architecture:

Campus Planning Inputs
        ↓
Feature Engineering
        ↓
Machine Learning Demand Forecast
        ↓
Predicted Food Demand
        ↓
Prepared vs Predicted Comparison
        ↓
Potential Surplus + Risk
        ↓
Preparation Recommendation
        ↓
Waste Analytics + Sustainability Indicators
        ↓
RAG Knowledge Retrieval
        ↓
Local LLM AI Assistant
        ↓
Human Review & Decision

🧠 AI Components

1. Machine Learning

The demand prediction layer uses a trained machine-learning model to estimate food consumption.

Input features include:

Number of students

Event indicator

Special-menu indicator

Day number

Menu type

The model output is an estimated number of meals expected to be consumed.

2. Surplus Detection

The application compares food prepared with predicted demand.

Potential Surplus =
Food Prepared − Predicted Demand

A positive value represents a potential planning surplus.

It is an estimate, not measured food waste.

3. Risk Classification

FoodWise classifies the potential surplus into planning-risk categories:

Potential Surplus

Classification

> 25 meals

🔴 High

> 10 meals

🟠 Medium

> 0 meals

🟡 Low

≤ 0 meals

🟢 No Surplus

These thresholds are prototype decision-support rules.

4. Preparation Recommendation

FoodWise calculates a recommended preparation quantity using the predicted demand and an operational buffer.

Current prototype buffer:

5%

Recommended Preparation =
Predicted Demand × (1 + Operational Buffer)

📊 Waste Analytics

The Sustainability Intelligence dashboard provides:

Total meals prepared

Total meals consumed

Total meals wasted

Historical waste rate

Prepared/consumed/wasted trends

Waste composition

Daily waste-rate visualization

Current scenario vs historical baseline

Waste Rate

Waste Rate =
Meals Wasted ÷ Meals Prepared × 100

Historical analytics are calculated from the project's dataset.

🌍 Sustainability Impact

FoodWise provides contextual sustainability indicators to demonstrate how avoided surplus could be interpreted.

Current prototype assumptions:

Illustrative food value: ₹45 per meal

Illustrative CO₂e factor: 0.4 kg CO₂e per meal

These are demonstration assumptions, not measured financial savings or verified avoided emissions.

The application explicitly labels these figures to avoid presenting estimates as real-world impact measurements.

🧠 RAG + Local LLM

FoodWise includes a Retrieval-Augmented Generation workflow.

Knowledge Base

The project knowledge base contains documents covering:

SDG 12

Food waste

Redistribution

Responsible AI

FoodWise system information

Retrieval

The application uses:

TF-IDF + cosine similarity

to retrieve relevant knowledge passages for a user question.

Generation

A locally running LLM can then generate a contextual response using the retrieved knowledge and current FoodWise scenario data.

Configured prototype model:

Qwen 2.5 1.5B

The local setup helps keep the prototype lightweight and avoids requiring a paid hosted LLM API.

🤖 FoodWise AI Copilot

The AI Assistant can answer questions such as:

How can FoodWise reduce food waste?

What is SDG 12.3?

How much food should we prepare?

What should we do with potential surplus?

The assistant separates:

Application-calculated values

from

AI-generated explanations

This helps reduce numerical hallucination and makes the system easier to audit.

🛡️ Responsible AI

FoodWise includes a dedicated Trust Center covering:

🔐 Data Privacy

Use only the data needed for the planning use case and prefer aggregate operational information over personally identifiable student information.

⚖️ Fairness & Bias

Historical data may not represent every future campus condition. Model performance should be reviewed across relevant conditions as better operational data becomes available.

🧠 Explainability

The application exposes:

Input conditions

Predicted demand

Prepared quantity

Potential surplus

Risk level

Recommended preparation

Retrieved knowledge

Model evaluation information

👤 Human Oversight

FoodWise provides decision support. Authorized staff remain responsible for operational decisions.

🍱 Food Safety

FoodWise does not determine whether food is safe to serve or redistribute.

Real redistribution workflows require appropriate institutional procedures, storage controls, handling requirements, and human verification.

🌍 Sustainability Transparency

Illustrative food-value and CO₂e calculations are clearly identified as assumptions rather than measured outcomes.

🏗️ System Architecture

┌─────────────────────────────┐
│       Campus Data           │
│ Students / Menu / Events    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│     Feature Engineering     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│   ML Demand Prediction      │
│      Random Forest          │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Predicted Demand & Surplus  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Risk + Preparation Planning │
└──────────────┬──────────────┘
               ↓
       ┌───────┴────────┐
       ↓                ↓
┌──────────────┐ ┌───────────────┐
│ Waste        │ │ Sustainability│
│ Analytics    │ │ Impact        │
└──────────────┘ └───────────────┘

Knowledge Base
      ↓
 TF-IDF Retrieval
      ↓
 Local LLM
      ↓
 FoodWise AI Copilot
      ↓
 Streamlit Dashboard
      ↓
 Human Review

🖥️ Application Modules

Module

Purpose

🎯 Prediction Engine

Forecast demand and support preparation planning

📊 Waste Analytics

Explore historical waste patterns

🤖 AI Assistant

Answer FoodWise and sustainability questions

🧠 AI Explainability

Show model metrics and feature importance

🏗️ System Architecture

Explain the end-to-end technical workflow

🌍 Impact & Report

Summarize sustainability indicators and export a CSV

🛡️ Trust Center

Document Responsible AI principles and limitations

🛠️ Technology Stack

Programming

Python

Data Science

Pandas

NumPy

Scikit-learn

Joblib

Machine Learning

Random Forest

Feature engineering

Regression evaluation metrics

RAG

TF-IDF

Cosine similarity

Local project knowledge base

Generative AI

Ollama

Qwen 2.5 1.5B

Visualization

Plotly

Streamlit charts

Application

Streamlit

Development

VS Code

Git / GitHub

Python virtual environment

📁 Project Structure

FoodWise-AI-main/
│
├── FoodWise-AI-main/
│   ├── app.py
│   ├── food_data.csv
│   ├── food_prediction_model.pkl
│   ├── menu_encoder.pkl
│   ├── model_comparison.pkl
│   ├── rag_engine.py
│   ├── llm_engine.py
│   │
│   └── knowledge_base/
│       ├── sdg12.txt
│       ├── food_waste.txt
│       ├── redistribution.txt
│       ├── responsible_ai.txt
│       └── foodwise_system.txt
│
└── .venv/

⚙️ Installation

1. Create / activate the virtual environment

From the project environment:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, you can run Streamlit using the environment's Python directly.

2. Install dependencies

pip install streamlit pandas numpy scikit-learn plotly joblib requests

3. Start FoodWise AI

streamlit run app.py

The application will open in your browser.

🤖 Optional Local AI Setup

FoodWise can use Ollama for local LLM responses.

Check Ollama:

ollama version

Download the configured model:

ollama pull qwen2.5:1.5b

Run the model:

ollama run qwen2.5:1.5b

Then return to the FoodWise application and use:

🤖 AI Assistant

🧪 Example Demo Flow

Scenario

Enter the number of students.

Select the menu type.

Set event/special-menu conditions.

Enter the amount of food prepared.

Run the prediction.

Review predicted demand.

Review potential surplus.

Check the risk classification.

Review the preparation recommendation.

Open Waste Analytics.

Ask the AI Copilot a sustainability question.

Open Explainability.

Open the Trust Center.

Open Impact & Report.

Download the sustainability CSV.

📈 Evaluation

The application displays common regression metrics:

R²

MAE

RMSE

MAPE

Important:

If the model metrics are calculated using the same dataset used for model development, they should not be presented as independent real-world test performance.

A future production version should use a clearly separated training, validation and held-out test methodology.

♻️ Redistribution Concept

FoodWise can support a redistribution-planning workflow when potential surplus is detected.

The prototype can present sample/demo redistribution options, but these records are not confirmed partner availability.

Any real redistribution action requires:

Human verification

Food-safety verification

Appropriate handling/storage checks

Confirmation of recipient availability

Authorized institutional approval

🎯 SDG Alignment

UN SDG 12 — Responsible Consumption and Production

FoodWise supports the project's sustainability objective through:

Demand forecasting

Food-preparation planning

Potential-surplus detection

Historical waste analytics

Sustainability reporting

Redistribution planning support

Target 12.3

The project focuses on reducing food waste and food losses through improved planning and decision support.

👥 Target Users

Potential users include:

Campus cafeterias

College canteens

Institutional kitchens

Food-service managers

Sustainability teams

Campus administrators

Student sustainability initiatives

🚧 Current Limitations

FoodWise is a prototype and has several limitations:

The project dataset may not represent every campus.

Current planning inputs are not a live institutional data pipeline.

Model evaluation requires stronger independent validation for production use.

Sample redistribution partners are demonstration records.

Sustainability impact factors are illustrative assumptions.

The LLM may produce incorrect language and requires review.

The application does not determine food safety.

Real-world savings and emissions reductions require validated measurements.

🔮 Future Enhancements

Possible future development includes:

Live cafeteria/attendance data integration

Time-series forecasting

Larger and more representative datasets

Independent model validation

Automated daily demand ingestion

Real partner/NGO verification workflows

Role-based access control

Database-backed historical analytics

Mobile-friendly operational interface

Cloud deployment

Advanced explainability methods

Automated sustainability reporting

Multilingual AI assistance

🌱 Expected Impact

FoodWise AI is intended to demonstrate how AI can support more informed food-preparation decisions and contribute to food-waste reduction efforts.

The prototype connects:

AI → Data → Decision Support → Sustainability

rather than treating AI as an autonomous decision-maker.

Its main value is demonstrating a practical AI + sustainability workflow aligned with UN SDG 12 and Target 12.3.

📜 Responsible Use

FoodWise AI should be treated as a prototype decision-support system.

Do not use its outputs as:

Proof of food safety

A replacement for institutional policies

A guarantee of food demand

Proof of financial savings

Proof of avoided emissions

Confirmation that a redistribution partner is available

Human review is required for real-world operational decisions.

📄 Project Deliverables

Recommended final submission package:

FoodWise-AI/
│
├── app.py
├── food_data.csv
├── food_prediction_model.pkl
├── menu_encoder.pkl
├── model_comparison.pkl
│
├── rag_engine.py
├── llm_engine.py
│
├── knowledge_base/
│   ├── sdg12.txt
│   ├── food_waste.txt
│   ├── redistribution.txt
│   ├── responsible_ai.txt
│   └── foodwise_system.txt
│
├── README.md
├── Responsible_AI.md
├── Project_Report.md
├── Demo_Script.md
└── FoodWise_Sustainability_Report.csv

👨‍💻 Project

FoodWise AI — Smart Campus Food Sustainability System

AI-Powered Food Demand Forecasting, Waste Reduction & Sustainable Redistribution Platform

Primary SDG: UN SDG 12 — Responsible Consumption and Production
Target: 12.3 — Reduce food waste and food losses by 2030

🌱 FoodWise AI: Predict smarter. Waste less. Plan sustainably.