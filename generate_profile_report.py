import pandas as pd
import webbrowser
import os
from ydata_profiling import ProfileReport

def generate_profile_report(input_file="StudentScore.xls", output_file="Score_report.html"):
    # Generate a detailed data analysis report and open it in web browser
    try:
        print(f"📖 Loading data from {input_file}...")
        # Load dataset
        df = pd.read_csv(input_file)
        print(f"✅ Data loaded successfully! Rows: {len(df)}, Columns: {len(df.columns)}")
        
        print(f"\n📊 Generating detailed analysis report...")
        # Create profile report with explorative analysis
        profile = ProfileReport(
            df, 
            title="Student Scores Analysis Report",
            explorative=True
        )
        
        print(f"💾 Saving report to {output_file}...")
        # Export report to HTML file
        profile.to_file(output_file)
        print(f"✅ Report saved successfully!")
        
        print(f"\n🌐 Opening report in web browser...")
        # Get absolute path and open in default browser
        absolute_path = os.path.abspath(output_file)
        webbrowser.open(f'file://{absolute_path}')
        print(f"✅ Report opened in default browser!")
        print(f"📄 File path: {absolute_path}")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
        print(f"   Please ensure the file exists in the same directory")
    except ModuleNotFoundError:
        print(f"❌ Error: ydata-profiling is not installed")
        print(f"   Run: pip install ydata-profiling")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    generate_profile_report()
