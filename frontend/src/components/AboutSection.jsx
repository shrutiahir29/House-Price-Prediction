import React from "react";
import { Database, GitBranch, Cpu, Code2, Users, Layers, Award, CheckCircle } from "lucide-react";

export default function AboutSection() {
  const pillars = [
    {
      role: "Data Engineering & Baseline",
      focus: "Data Preprocessing, EDA & Linear Regression",
      deliverables: [
        "Dataset cleaning & imputation (80,000 properties)",
        "80:20 partition splits (train_indices.csv, test_indices.csv)",
        "Feature schema definition (12 numerical + 4 categorical)",
        "Linear Regression baseline pipeline (R²: 0.6657, MAE: 47.94L)",
      ],
      color: "border-sky-500",
      badge: "bg-sky-100 text-sky-800",
    },
    {
      role: "Model Serving & Backend API",
      focus: "Random Forest & FastAPI Backend",
      deliverables: [
        "Random Forest Regressor ensemble training",
        "Hyperparameter tuning (100 estimators, max depth 25)",
        "FastAPI high-throughput backend (endpoints: /predict, /health)",
        "Pipeline serialization (models/random_forest_model.joblib)",
      ],
      color: "border-emerald-500",
      badge: "bg-emerald-100 text-emerald-800",
    },
    {
      role: "Advanced ML & Frontend UI",
      focus: "XGBoost, Decision Tree & HouseWise Frontend",
      deliverables: [
        "Decision Tree Regressor pipeline (R²: 0.7646, MAE: 34.95L)",
        "XGBoost Gradient Boosting Regressor (Champion R²: 0.7896, MAE: 32.49L)",
        "Full HouseWise React + Vite + Tailwind CSS application",
        "Interactive 4-Model Comparative Analytics Dashboard",
      ],
      color: "border-amber-500",
      badge: "bg-amber-100 text-amber-800",
    },
  ];

  return (
    <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      
      {/* Intro */}
      <div className="text-center max-w-3xl mx-auto mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 text-white text-xs font-semibold mb-3">
          <Database className="w-3.5 h-3.5 text-amber-400" />
          <span>SYSTEM ARCHITECTURE & ENGINEERING PILLARS</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 font-display">
          About HouseWise
        </h2>
        <p className="text-sm sm:text-base text-slate-600 mt-2 leading-relaxed">
          HouseWise is an AI-powered valuation and benchmarking system trained on 80,000 real estate properties across India's top 10 metropolitan regions. Built with strict data integrity across 3 core engineering modules.
        </p>
      </div>

      {/* Engineering Pillars Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {pillars.map((m) => (
          <div
            key={m.role}
            className={`bg-white rounded-3xl p-6 sm:p-7 border border-slate-200 shadow-lg border-t-4 ${m.color} flex flex-col justify-between`}
          >
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="text-base font-extrabold text-slate-900">{m.role}</span>
                <span className={`text-[11px] font-bold px-2.5 py-1 rounded-full ${m.badge}`}>
                  Active
                </span>
              </div>
              <h4 className="text-sm font-bold text-slate-800 mb-4 font-display">
                {m.focus}
              </h4>
              <ul className="space-y-2.5 text-xs text-slate-600">
                {m.deliverables.map((d, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                    <span>{d}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>

      {/* Dataset & Methodology */}
      <div className="bg-slate-900 text-white rounded-3xl p-8 sm:p-10 shadow-2xl relative overflow-hidden">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center md:text-left">
          <div>
            <span className="text-amber-400 text-xs font-bold uppercase tracking-wider block mb-1">
              DATA INTEGRITY
            </span>
            <h4 className="text-xl font-bold font-display text-white mb-2">
              80,000 Verified Homes
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Curated dataset representing Mumbai, Delhi, Bangalore, Pune, Hyderabad, Chennai, Kolkata, Ahmedabad, Jaipur, and Chandigarh.
            </p>
          </div>

          <div>
            <span className="text-amber-400 text-xs font-bold uppercase tracking-wider block mb-1">
              SCIENTIFIC SPLIT
            </span>
            <h4 className="text-xl font-bold font-display text-white mb-2">
              80:20 Partition Lock
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Exactly 64,000 train indices and 16,000 test indices were strictly locked across all 4 pipelines to prevent data leakage and guarantee unskewed comparative results.
            </p>
          </div>

          <div>
            <span className="text-amber-400 text-xs font-bold uppercase tracking-wider block mb-1">
              PRODUCTION ARCHITECTURE
            </span>
            <h4 className="text-xl font-bold font-display text-white mb-2">
              Full-Stack Monorepo
            </h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              FastAPI backend running Scikit-Learn and XGBoost joblib serialized pipelines served seamlessly to a modern React + Vite + Tailwind CSS client.
            </p>
          </div>
        </div>
      </div>

    </section>
  );
}
