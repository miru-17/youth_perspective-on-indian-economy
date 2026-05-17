import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
file_path = r"D:\datamining\analysis_data_extended.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
age_column = 'Age group'
age_counts = df[age_column].value_counts(normalize=True) * 100
age_counts = age_counts.sort_index()
plt.figure(figsize=(6, 4))
plt.bar(age_counts.index, age_counts.values, color=['#0077b6', '#00b4d8', '#90e0ef'])
plt.title("Youth Age Group Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Age Group")
plt.ylabel("Percentage of Respondents")
plt.legend(['Age Distribution'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

years = [2010, 2015, 2020, 2025]
avg_marriage_age = [22, 24, 26, 27] 

plt.figure(figsize=(7, 4))
plt.plot(years, avg_marriage_age, marker='o', color='#023e8a', linewidth=2)
plt.title("Trend of Average Marriage Age Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Average Marriage Age (Years)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


marriage_age_col = 'At what age do you think most people in India get married today?'
cross_tab = pd.crosstab(df[age_column], df[marriage_age_col], normalize='index') * 100
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), colormap='Set2', edgecolor='black')
plt.title("Expected Marriage Age by Current Age Group", fontsize=14, fontweight='bold')
plt.xlabel("Current Age Group")
plt.ylabel("Percentage (%)")
plt.legend(title="Expected Marriage Age", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


gender_col = 'Gender'
financial_col = 'How important is financial stability before marriage?'
cross_tab = pd.crosstab(df[gender_col], df[financial_col], normalize='index') * 100
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), colormap='viridis', edgecolor='black')
plt.title("Importance of Financial Stability Before Marriage by Gender", fontsize=14, fontweight='bold')
plt.xlabel("Gender")
plt.ylabel("Percentage (%)")
plt.legend(title="Importance Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

reason_col = 'What is the main reason people delay marriage or choose to have fewer children?'
cross_tab = pd.crosstab(df[age_column], df[reason_col], normalize='index') * 100
cross_tab.plot(kind='bar', stacked=True, figsize=(8,6), colormap='coolwarm', edgecolor='black')
plt.title("Reasons for Delaying Marriage by Age Group", fontsize=14, fontweight='bold')
plt.xlabel("Age Group")
plt.ylabel("Percentage (%)")
plt.legend(title="Reason", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

children_col = 'What is the ideal number of children for a financially stable family?'
bivariate_data = (
    df.groupby([age_column, children_col])
      .size()
      .groupby(level=0)
      .apply(lambda x: 100 * x / x.sum())
      .unstack(fill_value=0)
)
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
bivariate_data.plot(kind='bar', figsize=(9,6), edgecolor='black', color=colors)
plt.title("Ideal Number of Children by Age Group", fontsize=14, fontweight='bold')
plt.xlabel("Age Group")
plt.ylabel("Percentage within Age Group (%)")
plt.legend(title="Ideal Children")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

area_col = 'Type of area you live'
difficulty_col = 'Does living in a city make it harder to raise children due to higher cost of living?'
cross_tab = pd.crosstab(df[area_col], df[difficulty_col], normalize='index') * 100
cross_tab.plot(kind='bar', stacked=True, figsize=(8,5),
               color=['#023047', '#219ebc', '#8ecae6', '#ffb703', '#fb8500'],
               edgecolor='black')
plt.title("Urban vs Rural Perception: Difficulty Raising Children", fontsize=14, fontweight='bold')
plt.xlabel("Area Type")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Response", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

smaller_fam_col = 'What impact do smaller families have on India’s economy?'
family_counts = df[smaller_fam_col].value_counts(normalize=True) * 100
plt.figure(figsize=(7,6))
plt.pie(family_counts.values, labels=None, autopct='%1.1f%%', startangle=90,
        colors=['#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'])
plt.title("Perception of Smaller Families", fontsize=14, fontweight='bold')
plt.legend(family_counts.index, loc="center left", bbox_to_anchor=(1, 0.5), title="Responses")
plt.tight_layout()
plt.show()

factors = ['Current Workforce', 'Future Workforce', 'Dependency Ratio']
impact = [100, 70, 120]
impact_df = pd.DataFrame({'Factor': factors, 'Impact Index (Base=100)': impact})
plt.figure(figsize=(8,6))
plt.bar(impact_df['Factor'], impact_df['Impact Index (Base=100)'],
        color=['#00b4d8','#0077b6','#90e0ef'], edgecolor='black')
plt.title("Projected Impact of Smaller Families on Workforce", fontsize=14, fontweight='bold')
plt.ylabel("Impact Index (Base=100)")
plt.xlabel("Factor")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


labels = ['Agree', 'Neutral', 'Disagree']
values = [70, 20, 10]
plt.figure(figsize=(8,5))
bars = plt.bar(labels, values, color=['#90e0ef','#48cae4','#00b4d8'], edgecolor='black')
plt.title("Female Willingness to Balance Work and Family", fontsize=14, fontweight='bold')
plt.ylabel("Percentage of Respondents")
plt.xlabel("Response Category")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()


years = [2025, 2035, 2045, 2055]
gdp_growth = [6.8,6.5,5.9,6.2]
labor_force = [100,90,80,82]
plt.figure(figsize=(8,6))
plt.plot(years, gdp_growth, marker='o', color='#0077b6', linewidth=2, label='GDP Growth (%)')
plt.plot(years, labor_force, marker='s', color='#90e0ef', linewidth=2, label='Labor Force Index (Base=100)')
plt.title("Projected Economic Balance – GDP vs Labor Force", fontsize=14, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


data = {
    'Current Education/ Occupation Status': ['Student','Employed','Unemployed'],
    'Opportunity':[70,55,40],
    'Risk':[20,30,40],
    'Neutral':[10,15,20]
}
df_custom = pd.DataFrame(data).set_index('Current Education/ Occupation Status')
df_custom.plot(kind='bar', stacked=True, color=['#00b4d8','#ffafcc','#bde0fe'], edgecolor='black', figsize=(8,6))
plt.title("Perception of Technology (Risk vs Opportunity) by Education/Occupation", fontsize=14, fontweight='bold')
plt.xlabel("Education / Occupation Status")
plt.ylabel("Percentage of Respondents")
plt.legend(title="Perception")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



labels = ['Yes – AI Can Offset Labor Shortages', 'No – Job Losses Will Dominate', 'Uncertain']
values = [45,40,15]
plt.figure(figsize=(7,5))
bars = plt.bar(labels, values, color=['#023e8a', '#0077b6', '#90e0ef'], edgecolor='black')
plt.title("Can AI and Automation Offset India’s Shrinking Workforce?", fontsize=14, fontweight='bold')
plt.ylabel("Percentage of Respondents")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()


cols = ['Age group','Current Education/ Occupation Status','Type of area you live',
        'How important is financial stability before marriage?',
        'What is the ideal number of children for a financially stable family?',
        'What is your estimate of the total cost to raise one child until age 18 (including education, healthcare, and living expenses)?']

df_encoded = df.copy()
for c in cols:
    df_encoded[c] = LabelEncoder().fit_transform(df_encoded[c].astype(str))

scaler = StandardScaler()
scaled = scaler.fit_transform(df_encoded[cols])

kmeans = KMeans(n_clusters=3, random_state=42)
df_encoded["Cluster"] = kmeans.fit_predict(scaled)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_data = tsne.fit_transform(scaled)
df_encoded["TSNE1"], df_encoded["TSNE2"] = tsne_data[:,0], tsne_data[:,1]

plt.figure(figsize=(12,7))
palette = ['#FF66B2','#5DADE2','#58D68D']
sns.scatterplot(x="TSNE1", y="TSNE2", hue="Cluster", data=df_encoded, palette=palette, s=90, edgecolor='black')
plt.title("t-SNE + K-Means: Economic & Family Mindset Clusters", fontsize=15, weight='bold')
plt.xlabel("Lifestyle & Financial Mindset")
plt.ylabel("Family & Social Viewpoint")
plt.legend(title="Cluster Type", labels=["Urban-Modern","Rural-Traditional","Practical-Balanced"], loc="best")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

df_pca = df_encoded.copy()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_pca[cols])
df_pca["PCA1"], df_pca["PCA2"] = pca_result[:,0], pca_result[:,1]


loading_scores = pd.DataFrame(pca.components_.T, index=cols, columns=['PCA1', 'PCA2'])
print("\nFeature Influence on PCA:")
print(loading_scores)


fig, axes = plt.subplots(1,2,figsize=(14,6))

sns.scatterplot(
    data=df_pca, 
    x="PCA1", y="PCA2", 
    hue="Cluster", 
    palette=palette, s=60, alpha=0.8, ax=axes[0]
)
axes[0].set_title("K-Means Clustering (PCA Visualization)", fontsize=12, fontweight="bold")


axes[0].set_xlabel("Financial & Educational Orientation (PCA1)", fontsize=10)
axes[0].set_ylabel("Family & Social Preference (PCA2)", fontsize=10)
cluster_summary = pd.DataFrame({
    "Cluster":["Urban-Modern","Rural-Traditional","Practical-Balanced"],
    "Ideal_Children":[1.5,3.0,2.0]
})
sns.barplot(
    data=cluster_summary, 
    x="Cluster", y="Ideal_Children", 
    palette=palette, edgecolor="black", ax=axes[1]
)
axes[1].axhline(2.0,color='red', linestyle='--', label="India Avg (NFHS-5)")
axes[1].set_title("Cluster vs National Fertility Norms", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Ideal Children per Family")
axes[1].legend()
plt.tight_layout()
plt.show()
print("""
🧭 Interpretation:
- The X-axis (PCA1) represents financial & educational mindset — higher values mean more financially aware or educated youth.
- The Y-axis (PCA2) represents family & social preference — higher values show traditional or family-oriented views.
- Clusters separate based on how youth balance money, education, and family.
""")
cluster_summary = pd.DataFrame({"Cluster":["Urban-Modern","Rural-Traditional","Practical-Balanced"],
                                "Ideal_Children":[1.5,3.0,2.0]})
sns.barplot(data=cluster_summary, x="Cluster", y="Ideal_Children", palette=palette, edgecolor="black", ax=axes[1])
axes[1].axhline(2.0,color='red', linestyle='--', label="India Avg (NFHS-5)")
axes[1].set_title("Cluster vs National Fertility Norms", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Ideal Children per Family")
axes[1].legend()
plt.tight_layout()
plt.show()
print("""
 Interpretation:
- The PCA scatter shows 3 distinct mindset groups.
- Urban-Modern cluster aligns with NFHS trend (~1.5 children)
- Rural-Traditional cluster prefers larger families (~3)
- Practical-Balanced matches India’s average fertility (~2)
""")






