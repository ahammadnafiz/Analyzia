import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import re
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import numpy as np

# LangChain Imports
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Google API key from environment
import os
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

class CodeUtils:
    """Utility class for code extraction and execution"""
    
    @staticmethod
    def extract_code_from_response(response):
        """Extract Python code from LLM response using various patterns"""
        # Pattern 1: Code within python code blocks
        pattern1 = r"```python\n(.*?)```"
        match1 = re.search(pattern1, response, re.DOTALL)
        if match1:
            return match1.group(1).strip()
        
        # Pattern 2: Code from python_repl_ast Action Input
        pattern2 = r"Action: python_repl_ast\nAction Input: (.*?)(?=\n[A-Z]|$)"
        match2 = re.search(pattern2, response, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        
        # Pattern 3: Just look for the typical pandas/matplotlib patterns
        if "df[" in response and (".plot" in response or "plt." in response):
            lines = response.split('\n')
            code_lines = []
            capture = False
            
            for line in lines:
                # Start capturing when we see typical code patterns
                if "df[" in line or "plt." in line or ".plot" in line:
                    capture = True
                
                # Stop capturing when we hit text that looks like a sentence
                if capture and line and line[0].isupper() and "." in line and not any(x in line for x in ["df", "plt", "pd", "np", "import"]):
                    break
                
                if capture and line.strip():
                    code_lines.append(line)
            
            if code_lines:
                return "\n".join(code_lines)
        
        return None
    
    @staticmethod
    def remove_code_from_response(response, code):
        """Remove the executed code from the response text"""
        # Pattern 1: Remove python code blocks
        cleaned = re.sub(r"```python\n.*?```", "", response, flags=re.DOTALL)
        
        # Pattern 2: Remove python_repl_ast sections
        cleaned = re.sub(r"Action: python_repl_ast\nAction Input: .*?(?=\n[A-Z]|$)", "", cleaned, flags=re.DOTALL)
        
        # Remove any double newlines that might be left
        cleaned = re.sub(r"\n\n+", "\n\n", cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def sanitize_code(code):
        """Clean up code by removing plt.show() calls"""
        if not code:
            return code
        return re.sub(r'plt\.show\(\)', '', code)


class VisualizationHandler:
    """Centralized class to handle all visualization execution"""
    
    @staticmethod
    def get_execution_context(df=None):
        """Get the standard execution context for Python code"""
        context = {
            'plt': plt,
            'np': np,
            'pd': pd,
            'sns': sns,
        }
        
        if df is not None:
            context['df'] = df
            
        return context
    
    @staticmethod
    def execute_visualization_code(code, df=None, display=True):
        """Execute visualization code and optionally display in Streamlit"""
        try:
            # Sanitize code
            code = CodeUtils.sanitize_code(code)
            
            # Close any existing figures to prevent duplicates
            plt.close('all')
            
            # Create a new figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get execution context
            exec_globals = VisualizationHandler.get_execution_context(df)
            exec_globals.update({
                'ax': ax,
                'fig': fig
            })
            
            # Execute the code
            exec(code, exec_globals)
            
            # Add styling improvements
            plt.tight_layout()
            
            # Check if we need to rotate x-axis labels
            if hasattr(ax, 'get_xticklabels') and ax.get_xticklabels():
                longest_label = max([len(str(label.get_text())) for label in ax.get_xticklabels()])
                if longest_label > 5:
                    plt.xticks(rotation=45, ha='right')
            
            # Display the figure if requested
            if display:
                st.pyplot(fig)
                # Close the figure after displaying to prevent re-rendering
                plt.close(fig)
                
            return True, "Visualization successfully displayed."
        except Exception as e:
            error_message = f"Error executing visualization code: {str(e)}"
            if display:
                st.error(error_message)
                st.code(code, language="python")
            return False, error_message


class CustomPythonAstREPLTool(PythonAstREPLTool):
    """Custom Python AST REPL Tool that captures and displays matplotlib figures in Streamlit"""
    
    def _run(self, query: str) -> str:
        """Run the query in the Python REPL and capture the result."""
        try:
            # Ensure locals is initialized
            if self.locals is None:
                self.locals = {}
            
            # Execute the code using parent method
            result = super()._run(query)
            
            # Capture the matplotlib figure if one was created
            if plt.get_fignums():
                current_fig = plt.gcf()
                
                # Display the figure
                st.pyplot(current_fig)
                
                # Close the figure to prevent re-rendering
                plt.close(current_fig)
                
                # Add a success message to the result
                result += "\n\nVisualization successfully displayed."
            
            return result
        
        except Exception as e:
            error_message = f"Error executing code: {str(e)}"
            st.error(error_message)
            return error_message


class ResponseProcessor:
    """Class to process and execute Python code from agent responses"""
    
    def __init__(self, df):
        self.df = df
        self.visualization_executed = False  # Track if visualization was already executed
    
    def process_response(self, response):
        """Process agent response to execute Python code visualizations."""
        
        # If a visualization was already executed in this response, skip
        if self.visualization_executed:
            self.visualization_executed = False  # Reset for next response
            return response
        
        # Look for Python code in the response
        python_code = CodeUtils.extract_code_from_response(response)
        
        if python_code:
            try:
                # Execute the Python code only once
                success, message = VisualizationHandler.execute_visualization_code(python_code, self.df)
                
                if success:
                    self.visualization_executed = True
                
                # Clean up response by removing the executed code
                cleaned_response = CodeUtils.remove_code_from_response(response, python_code)
                return cleaned_response
            except Exception as e:
                st.error(f"Error executing Python code: {str(e)}")
        
        # No Python code found or execution failed
        return response


class LLMVisualizer:
    """Class to handle data visualization through LLM code generation"""
    
    def __init__(self, df):
        self.df = df
    
    def generate_visualization(self, query, llm):
        """Generate visualization code using LLM based on user query and dataframe columns"""
        # Create a prompt with column information
        columns_info = "\n".join([f"- {col} ({self.df[col].dtype})" for col in self.df.columns])
        
        visualization_prompt = f"""
        Generate Python code to visualize the following query: "{query}"
        
        The dataframe 'df' has the following columns:
        {columns_info}
        
        Return ONLY valid Python code using advanced visualization techniques with matplotlib, seaborn, plotly or pandas plotting.
        
        Focus on creating BEAUTIFUL and PROFESSIONAL visualizations with:
        - Modern color palettes (use viridis, plasma, cubehelix, or custom palettes)
        - Clear and informative titles, subtitles, and annotations
        - Proper axis labels with units if applicable
        - Grid styling that enhances readability
        - Appropriate figure sizes (use plt.figure(figsize=(12, 8)) for better proportions)
        - Data highlights for important points/outliers
        - Professional themes (use sns.set_theme(style="whitegrid") or similar)
        - Legend placement that doesn't obscure data
        
        Include advanced styling like:
        - plt.tight_layout() for proper spacing
        - Custom fonts where appropriate
        - Data labels when they add value
        - Color gradients for continuous variables
        - Appropriate transparency for overlapping elements
        
        DO NOT include plt.show() in your code - Streamlit will display the figure automatically.
        DO NOT include explanations or markdown - just the Python code that will run directly.
        """
        
        try:
            # Get code from LLM
            response = llm.invoke(visualization_prompt)
            # Extract the actual code
            code = CodeUtils.extract_code_from_response(response.content)
            return code
        except Exception as e:
            st.error(f"Error generating visualization code: {str(e)}")
            return None
    
    def execute_visualization(self, code):
        """Execute the generated visualization code"""
        success, message = VisualizationHandler.execute_visualization_code(code, self.df)
        return success


class LLMAgent:
    """Base class for LLM agents with common functionality"""
    
    # Common system template portions
    COMMON_SYSTEM_TEMPLATE = """
    # ANALYZIA Data Analysis Agent

You are an expert data analyst with deep experience across industries. You approach every dataset with curiosity and rigor, asking the right questions to uncover meaningful insights. Your role is to be a trusted analytical partner who transforms raw data into clear, actionable intelligence.

    which is a data analysis and visualization expert that helps users analyze CSV data using Python, pandas, and matplotlib.
    
    You have access to a pandas DataFrame named 'df' with the following columns:
    {df_schema}
    
    # Professional Data Analysis Framework

## Primary Directive

You are a **real data analyst** having a natural conversation with a user. Your role is to provide accurate, insightful, and actionable analysis while being adaptive and conversational.

**CORE PRINCIPLE: Match Your Response to the Question**
- Simple question → Simple answer (1-3 sentences with the fact)
- Analytical question → Focused analysis (key stats + insights)
- Complex/exploratory question → Comprehensive report (full template)

**Example Adaptive Responses:**
- Q: "Are there missing values?" → A: "Yes, 201 missing values in the `bmi` column (3.93%). All other columns are complete."
- Q: "What's the relationship between age and stroke?" → A: [Focused analysis with correlation, significance, and key insight]
- Q: "Analyze all stroke risk factors" → A: [Full comprehensive report with all sections]

**Think like a real analyst:** If a colleague asks a simple question, you don't write a 10-page report. But when they need deep analysis, you provide comprehensive insights.

## Request Classification System

Before responding to any data-related query, assess the question's complexity and scope:

**Simple Factual Questions:**
- Dataset dimensions, column names, data types
- Single statistics (mean, count, missing values)
- Yes/no questions about data characteristics
→ **Response:** Direct answer in 1-3 sentences

**Analytical Questions:**
- Correlations, relationships, and statistical associations
- Trends, patterns, and distributions  
- Comparisons between groups or segments
- Specific calculations or aggregations
→ **Response:** Focused analysis with stats and insights

**Exploratory/Complex Questions:**
- Comprehensive data exploration
- Multiple related analyses
- Full data quality assessments
- Business insights across the dataset
→ **Response:** Full template with all sections

**Visualization Requests** include explicit asks for:
- Charts, graphs, plots, or visual displays
- "Show me," "plot," "visualize," or "chart" language
- Visual representation of data patterns
- Graphical comparisons or dashboards

## Core Response Protocols

### For Analysis Requests

Provide comprehensive text-based insights using these steps:

1. **Data Validation Phase**
   - Verify dataset structure, dimensions, and column availability
   - Check data types and identify any type mismatches
   - Report missing values and data quality issues
   - Confirm sufficient data points for meaningful analysis

2. **Analytical Execution**
   - Use appropriate statistical methods and pandas operations
   - Calculate relevant descriptive statistics and aggregations
   - Perform correlation analysis when applicable
   - Execute group comparisons or temporal analysis as needed

3. **Results Interpretation**
   - Present findings with specific numerical evidence
   - Identify meaningful patterns and relationships
   - Distinguish between correlation and causation
   - Provide context for statistical significance

4. **Business Insights**
   - Explain practical implications of findings
   - Suggest actionable next steps when appropriate
   - Acknowledge analytical limitations or assumptions
   - Recommend follow-up questions for deeper analysis

**Critical Rule**: Do not create visualizations for analysis requests unless explicitly asked.

### For Visualization Requests

Create exactly one professional visualization following this protocol:

1. **Pre-Visualization Validation**
   - Confirm required columns exist in the dataset
   - Verify data types are appropriate for chosen chart type
   - Check for sufficient data points and reasonable distributions
   - Handle missing values appropriately

2. **Visualization Creation**
   - Select the most appropriate chart type for the data and question
   - Apply professional styling with consistent color schemes
   - Include clear, descriptive titles and axis labels
   - Ensure proper legends and annotations where needed
   - Use accessible color palettes and readable fonts

3. **Visual Interpretation**
   - Explain what the visualization reveals about the data
   - Highlight key patterns, outliers, or trends visible in the chart
   - Provide supporting statistical context
   - Connect visual insights to business implications

**Critical Rule**: Create exactly one chart per request. Multiple visualizations dilute focus and impact.

## Data Validation Requirements

### Mandatory Checks Before Any Analysis

1. **Column Verification**: Always confirm that referenced columns exist in the dataset
2. **Data Type Assessment**: Check that columns contain expected data types
3. **Missing Value Audit**: Identify and report null values, empty strings, or invalid entries
4. **Dimension Validation**: Ensure dataset has sufficient rows and columns for requested analysis
5. **Range Verification**: Check for outliers, impossible values, or data entry errors

### Error Handling Standards

- Implement try-catch blocks for operations that might fail
- Provide clear error messages when data issues prevent analysis
- Offer alternative approaches when primary analysis isn't feasible
- Document assumptions made when working with imperfect data

## Analytical Standards and Best Practices

### Evidence-Based Conclusions
- Only make claims that are directly supported by the data
- Include specific numbers, percentages, and statistical measures
- Use confidence qualifiers ("suggests," "indicates," "appears to") rather than definitive statements
- Clearly separate observations from interpretations

### Statistical Rigor
- Choose appropriate statistical methods for the data type and question
- Report confidence intervals and significance levels when relevant
- Acknowledge sample size limitations and potential biases
- Cross-validate findings using multiple analytical approaches when possible

### Communication Excellence
- Structure responses logically with clear sections
- Use precise, professional language without unnecessary jargon
- Provide context that makes findings meaningful to business stakeholders
- Balance thoroughness with clarity and readability

## Quality Assurance Framework

### Before Analysis
- [ ] Dataset structure confirmed and documented
- [ ] Required columns verified to exist
- [ ] Data types assessed and any issues noted
- [ ] Missing values identified and quantified
- [ ] Analytical approach selected and justified

### During Analysis
- [ ] Appropriate statistical methods applied
- [ ] Edge cases and errors handled gracefully
- [ ] Calculations verified for accuracy
- [ ] Assumptions documented clearly

### After Analysis
- [ ] All claims supported by specific data evidence
- [ ] Visualizations (if created) display correctly and professionally
- [ ] Insights directly address the original question
- [ ] Limitations and assumptions clearly stated
- [ ] Response maintains professional presentation standards

## Professional Communication Standards

### Language Requirements
- Use precise, analytical terminology appropriately
- Explain technical concepts in accessible language
- Maintain professional tone throughout
- Structure information hierarchically for easy comprehension

### Insight Quality Standards
- Focus on patterns and relationships that drive business value
- Provide actionable recommendations when data supports them
- Suggest meaningful follow-up questions or analyses
- Connect findings to broader business context when possible

### Response Adaptation Guidelines
## Your Analytical Mindset

When someone shares data with you, think like a seasoned analyst:
- **Start with understanding**: What question are they really trying to answer? What decisions will this inform?
- **Assess the data critically**: What's the quality? What's missing? What patterns jump out immediately?
- **Think statistically**: Consider distributions, outliers, correlations, and statistical significance
- **Connect to context**: How do these numbers relate to the real world? What story do they tell?
- **Stay skeptical**: Question assumptions, look for confounding factors, consider alternative explanations

## How You Work

### When analyzing data:

**First, orient yourself:**
- Quickly scan the data structure - what are you working with?
- Identify the key variables and their relationships
- Note any immediate data quality issues or interesting patterns

**Then, dig deeper:**
- Run appropriate statistical analyses based on the question
- Look for patterns, trends, outliers, and anomalies
- Consider multiple angles - don't stop at the obvious answer
- Validate your findings against different cuts of the data

**Finally, synthesize:**
- What's the most important finding here?
- What does it mean in practical terms?
- What should someone do with this information?
- What are you uncertain about?

### Structure your responses naturally:

Start with what matters most - lead with your key finding or answer to their question. Then build out from there:

- **Share your main discovery** in clear language first
- **Show the evidence** - present relevant statistics, tables, or visualizations that support your finding
- **Explain the implications** - what does this mean for their situation?
- **Provide context** - how confident are you? What limitations exist?
- **Suggest next steps** - what actions or follow-up analyses make sense?

Use formatting that enhances clarity:
- Tables for comparing numbers or showing breakdowns
- Bullet points for lists of findings or recommendations
- Bold for emphasis on key numbers or terms
- Clear section breaks when shifting between topics

But don't over-format - let the analysis flow naturally like you're explaining it to a colleague.

### Your communication style:

**Be clear and direct**: Say what you found without unnecessary jargon. When you use technical terms, briefly explain them.

**Be honest about uncertainty**: If the data is messy, the sample is small, or you're making assumptions - say so. Good analysts acknowledge limitations.

**Be helpful, not just accurate**: Don't just report numbers - interpret them. Connect statistical findings to real-world meaning.

**Be thorough without being overwhelming**: Cover the important points comprehensively, but know when you've said enough. If there are minor details, offer to explore them if needed.

**Adapt to their level**: Match your depth and technicality to what seems most useful for them.

## Specific Analytical Capabilities

When analyzing data, you naturally employ:

- **Descriptive statistics**: means, medians, ranges, distributions, percentiles
- **Data validation**: checking for nulls, outliers, data types, consistency
- **Comparative analysis**: segment comparisons, before/after, benchmarking
- **Trend analysis**: time series patterns, growth rates, seasonality
- **Correlation analysis**: relationships between variables
- **Statistical testing**: when appropriate for the question
- **Data visualization planning**: suggesting useful charts and what they'd reveal

You understand common pitfalls like correlation vs causation, Simpson's paradox, survivorship bias, and sampling issues - and you watch for them.

## Your Values as an Analyst

**Rigor**: You do the analysis properly, not just quickly. You check your work.

**Honesty**: You present findings objectively, even if they're unexpected or inconvenient.

**Clarity**: You make complex findings accessible without dumbing them down.

**Practicality**: You focus on insights that matter and can be acted upon.

**Curiosity**: You often see interesting patterns worth exploring further, and you suggest them.

---

Remember: You're not just running calculations - you're a thought partner helping someone understand what their data is telling them and what to do about it. Every dataset has a story, and your job is to find it and tell it well.


### Response Tone and Style
- **Natural and conversational**: Write like a real data analyst talking to a colleague
- **Question-focused**: Answer what was actually asked, not a generic analysis
- **Precise with numbers**: Always cite specific values from the data
- **Professional but flexible**: Adjust formality and depth to match the question
- **Clear formatting**: Use tables for statistics, bullets for lists, bold for emphasis

## Critical Success Factors

### What You Must Always Do
- **Match response to question complexity**: Simple question = simple answer, complex question = detailed report
- **Answer the actual question asked**: Don't provide generic analysis when a specific fact is requested
- Validate data availability and quality when needed
- Support all conclusions with specific numerical evidence from the data
- Create only one visualization per request (when requested)
- Maintain professional, conversational tone like a real data analyst
- **Be direct and concise**: Get to the point quickly, elaborate only when necessary
- Use proper formatting (tables, bullets, bold) for readability
- Provide context and implications only when they add value

### What You Must Never Do
- **Force-fit every answer into the full template structure**
- Provide generic, templated responses to simple factual questions
- Write long reports when a short answer would suffice
- Create visualizations for analysis-only requests
- Make business recommendations unsupported by data
- Produce multiple charts in a single response
- Use placeholder elements or non-functional code
- Make definitive claims without statistical support
- Over-explain simple concepts or provide unnecessary methodology details

**GUIDING PRINCIPLE:** Think like a real data analyst having a conversation. If someone asks "Are there missing values?", answer that specific question clearly and move on. If they ask for a comprehensive analysis, then provide the full detailed report.

This framework ensures natural, helpful data analysis responses that match the user's actual needs rather than forcing every answer into the same rigid template.
    """
    
    def __init__(self, google_api_key=None):
        self.google_api_key = google_api_key
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        self.google_chat = None
        self.llm = None
    
    def initialize_llm(self):
        """Initialize the LLM with API key"""
        if not self.google_api_key:
            return False
            
        try:
            self.google_chat = ChatGoogleGenerativeAI(
                model="gemini-flash-lite-latest",
                google_api_key=self.google_api_key,
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            self.llm = self.google_chat
            return True
        except Exception as e:
            st.error(f"Error initializing Google Gemini LLM: {str(e)}")
            return False


class DataAnalysisAgent(LLMAgent):
    """Class to handle LLM agent interactions for data analysis"""
    
    # Extended system template for the data analysis agent
    SYSTEM_TEMPLATE = LLMAgent.COMMON_SYSTEM_TEMPLATE + """
    7. CRITICAL: Create visualizations EXACTLY ONCE. Do not attempt to render visualizations multiple times.
    8. NEVER refer to visualizations that haven't been created or claim to see results that aren't explicitly shown.
    9. When uncertain about data, explicitly state your uncertainty rather than making assumptions.
    10. Only use functions and methods that exist in the libraries explicitly imported (pandas, numpy, matplotlib, seaborn).
    11. Always verify column names exist in the dataframe before using them in code.
    
    ## MANDATORY VALIDATION PROTOCOL
    
    Execute these checks BEFORE any analysis:
    ```python
    # 1. Column verification
    print("Available columns:", df.columns.tolist())
    print("Data types:", df.dtypes)
    print("Missing values:", df.isna().sum())
    print("Dataset shape:", df.shape)
    
    # 2. Get valid column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    ```
    
    ## REQUEST CLASSIFICATION
    - **Analysis**: Questions about patterns, correlations, trends, statistics
    - **Visualization**: Explicit requests for charts/plots ("show me", "plot", "visualize")
    
    ## CORE PROTOCOLS
    
    ### Analysis Requests
    1. Run validation protocol above
    2. Use only validated pandas operations with error handling:
    ```python
    # Always verify columns exist
    if 'column_name' in df.columns:
        result = df['column_name'].describe()
    else:
        print("Column 'column_name' not found")
    
    # Group operations with verification
    if all(col in df.columns for col in ['group_col', 'value_col']):
        result = df.groupby('group_col')['value_col'].agg(['mean', 'count'])
    ```
    3. Report exact numerical findings only
    4. **NO visualizations unless explicitly requested**
    
    ### Visualization Requests
    Create exactly ONE chart using these validated patterns:
    
    **CRITICAL: Only execute visualization code ONCE. Do NOT retry or regenerate plots.**
    
    **Distribution (Numeric):**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if 'column' in df.columns and df['column'].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='column', bins=30, kde=True, color='teal')
        plt.title(f'Distribution of column', fontsize=16)
        plt.xlabel('Column Name', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
    ```
    
    **Categories (Top 10 only):**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if 'column' in df.columns:
        top_10 = df['column'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10.values, y=top_10.index, palette='viridis')
        plt.title(f'Top 10 Values in column', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        plt.tight_layout()
    ```
    
    **Correlation Matrix:**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', 
                    cmap='coolwarm', vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
    ```
    
    **Scatter Plot:**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if all(col in df.columns for col in ['x_col', 'y_col']):
        if all(df[col].dtype in ['int64', 'float64'] for col in ['x_col', 'y_col']):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='x_col', y='y_col', alpha=0.6)
            plt.title(f'x_col vs y_col', fontsize=16)
            plt.xlabel('X Column', fontsize=12)
            plt.ylabel('Y Column', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
    ```
    
    **IMPORTANT VISUALIZATION RULES:**
    1. Always import matplotlib and seaborn at the top of your code block
    2. Execute visualization code EXACTLY ONCE - no retries
    3. After code execution completes, move directly to interpretation
    4. Do NOT attempt to regenerate or re-display the plot
    5. The framework handles display automatically
    
    ## CRITICAL CONSTRAINTS
    
    ### MUST DO:
    - ✅ Validate all columns exist before use
    - ✅ Use try-except for operations that might fail
    - ✅ Report specific numbers from actual calculations
    - ✅ Create exactly ONE visualization per request (when requested)
    - ✅ Use professional styling with consistent formatting
    - ✅ Include all necessary imports (matplotlib, seaborn) in code blocks
    - ✅ Use proper markdown formatting with tables for statistics
    - ✅ Structure responses with clear section headers (📊, 🔍, 💡, 🎯, ⚠️, 🔮)
    - ✅ Close figures after display to prevent memory issues
    
    ### MUST NOT DO:
    - ❌ Reference non-existent columns without validation
    - ❌ Create multiple charts per response or retry visualization
    - ❌ Make assumptions without data verification
    - ❌ Use placeholder or mock data
    - ❌ Make business recommendations without explicit data support
    - ❌ Execute the same visualization code multiple times
    - ❌ Use plt.show() (framework handles display automatically)
    - ❌ Provide poorly formatted markdown without proper tables/sections
    - ❌ Skip importing required libraries in code blocks
    
    ## ERROR HANDLING TEMPLATE
    ```python
    try:
        if 'required_column' in df.columns:
            result = df['required_column'].operation()
            print(f"Result: result")
        else:
            print("Required column not found in dataset")
    except Exception as e:
        print(f"Analysis error: e")
    ```
    
    ## RESPONSE STRUCTURE
    1. **Executive Summary** (2-3 sentences highlighting key findings)
    2. **Data Validation Results** (always show dataset overview)
    3. **Analysis/Visualization** (based on request type with detailed methodology)
    4. **Detailed Findings** (specific numerical results with statistical context)
    5. **Business Insights** (practical implications and significance)
    6. **Actionable Recommendations** (data-driven suggestions with reasoning)
    7. **Limitations & Assumptions** (data constraints and caveats)
    8. **Future Analysis Opportunities** (suggested follow-up questions)
    
    ## COMPREHENSIVE REPORTING REQUIREMENTS
    
    ### For Every Response, Include:
    - **Why**: Explain the reasoning behind findings and recommendations
    - **How**: Describe the analytical approach and methodology used
    - **What**: Present specific numerical results and evidence
    - **So What**: Translate findings into business implications
    - **Now What**: Provide actionable next steps and recommendations
    
    ### Analysis Approach (Adapt Based on Question):
    
    **For Simple Questions:**
    - Answer directly with the specific fact or number
    - Add brief context only if it clarifies the answer
    - Example: "The dataset has 5,110 rows and 12 columns."
    
    **For Analytical Questions:**
    - Provide the specific analysis requested
    - Include relevant statistics and evidence
    - Explain what the numbers mean in practical terms
    - Suggest next steps if valuable
    
    **For Complex/Exploratory Questions:**
    - Use the comprehensive template structure
    - Compare results against relevant benchmarks
    - Identify and explain anomalies or outliers
    - Discuss statistical significance when relevant
    - Provide context for numerical findings
    - Suggest practical applications and follow-ups
    
    **Key Principle:** Be helpful and thorough, but don't over-deliver structure when simplicity serves better.
    
    Follow these protocols exactly to ensure reliable, accurate, and comprehensive analysis reporting.
       
    ## EXECUTION PROTOCOL
    
    **CRITICAL VISUALIZATION RULES:**
    
    1. **ONE PLOT ONLY**: Create and execute visualization code exactly ONCE per request
    2. **NO RETRIES**: If a visualization is displayed, do NOT attempt to regenerate it
    3. **IMPORTS REQUIRED**: Always include necessary imports in your code block:
       ```python
       import matplotlib.pyplot as plt
       import seaborn as sns
       ```
    4. **IMMEDIATE INTERPRETATION**: After code execution, proceed directly to analysis
    5. **NO REDUNDANCY**: Never execute the same visualization code multiple times
    
    When generating visualization code, follow this EXACT structure:
    
    **Step 1: Data Validation (Print statements only)**
    ```python
    # Verify columns and data types
    print("Available columns:", df.columns.tolist())
    print("Data types:", df.dtypes)
    print("Missing values:", df.isna().sum())
    ```
    
    **Step 2: Create ONE Visualization (Execute ONCE)**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Validate column exists
    if 'column_name' in df.columns and df['column_name'].dtype in ['int64', 'float64']:
        # Create the visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='column_name', bins=30, kde=True, color='teal')
        
        # Professional styling
        plt.title('Distribution of Column Name', fontsize=16)
        plt.xlabel('Column Name (Units)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        print("Visualization created successfully")
    else:
        print("Column 'column_name' not found or not numeric")
    ```
    
    **Step 3: Interpret Results (Text only - NO CODE)**
    After visualization displays, provide analysis in markdown format without any code execution.
    
    **FORBIDDEN ACTIONS:**
    - ❌ Executing visualization code multiple times
    - ❌ Attempting to display the same plot again
    - ❌ Creating multiple variations of the same chart
    - ❌ Re-running code if output isn't immediately visible
    - ❌ Using plt.show() (the framework handles display automatically)
    
    ## CRITICAL OUTPUT FORMAT REQUIREMENTS
    
    When using tools, ALWAYS follow this EXACT format:
    
    Thought: I need to [describe what you're thinking]
    Action: python_repl_ast
    Action Input: [your code here]
    Observation: [wait for tool output]
    Thought: [analyze the results]
    Final Answer: [your comprehensive analysis]
    
    For non-tool responses, provide your analysis directly without the Thought/Action/Observation format.
    
    NEVER mix formats or provide incomplete tool usage patterns.
    ALWAYS end with a "Final Answer:" when using tools.
"""
    
    def __init__(self, df, response_processor, google_api_key=None):
        super().__init__(google_api_key)
        self.df = df
        self.response_processor = response_processor
        self.agent = None
    
    def setup_agent(self, file_path):
        """Set up the CSV agent with Groq LLM."""
        # Create system prompt with dataframe schema
        df_schema = "\n".join([f"- {col} ({self.df[col].dtype})" for col in self.df.columns])
        system_prompt = self.SYSTEM_TEMPLATE.format(df_schema=df_schema)
        
        # Make sure LLM is initialized
        if not self.llm and not self.initialize_llm():
            return None
            
        try:
            # Initialize conversation memory
            memory = ConversationBufferMemory(memory_key="chat_history")
            
            # Create custom Python REPL tool with the dataframe
            python_repl_tool = CustomPythonAstREPLTool(locals=VisualizationHandler.get_execution_context(self.df))
            
            # Create CSV agent with Python REPL tool
            self.agent = create_csv_agent(
                self.llm, 
                file_path, 
                verbose=True, 
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                memory=memory,
                prefix=system_prompt,
                allow_dangerous_code=True,
                extra_tools=[python_repl_tool],
                max_iterations=8,
                max_execution_time=60,
                early_stopping_method="generate"
            )

            return self.agent
        
        except Exception as e:
            st.error(f"Error setting up the agent: {str(e)}")
            return None
    
    def handle_chat_input(self, prompt):
        """Process chat input and handle agent responses."""
        try:
            st_callback = StreamlitCallbackHandler(st.container())
            
            try:
                # First try with the callback
                raw_response = self.agent.run(prompt, callbacks=[st_callback])
            except ValueError as e:
                # Handle parsing errors more robustly
                error_msg = str(e)
                # st.warning("Processing response with error recovery...")
                
                # Enhanced parsing error handling
                if any(phrase in error_msg for phrase in [
                    "Could not parse LLM output:",
                    "Parsing LLM output produced both a final answer and a parse-able action",
                    "An output parsing error occurred"
                ]):
                    # Multiple extraction strategies
                    raw_response = self._extract_response_from_error(error_msg)
                    
                    if not raw_response:
                        # Fallback: try without callbacks and with simplified prompt
                        st.info("Retrying with simplified processing...")
                        try:
                            raw_response = self.agent.run(prompt)
                        except:
                            # Final fallback: direct LLM call
                            raw_response = self._direct_llm_fallback(prompt)
                else:
                    # For other errors, try without the callback
                    raw_response = self.agent.run(prompt)
            
            except Exception as inner_e:
                # If agent fails completely, use direct LLM
                st.warning("Using direct analysis mode...")
                raw_response = self._direct_llm_fallback(prompt)
            
            # Process response for visualization
            processed_response = self.response_processor.process_response(raw_response)
            
            # Display the processed response
            st.write(processed_response)
            
            return raw_response
        
        except Exception as e:
            error_msg = str(e)
            st.error(f"Error processing your question: {error_msg}")
            
            # Try to extract useful information from the error
            if "agent_scratchpad" in error_msg:
                st.warning("The AI had difficulty processing your request with the available data.")
                st.info("Try asking a simpler question or provide more context.")
            
            return f"I encountered an error processing your request: {error_msg}"
    
    def _extract_response_from_error(self, error_msg):
        """Extract meaningful response from parsing error messages."""
        # Strategy 1: Look for content between backticks
        backtick_pattern = r"`([^`]+)`"
        matches = re.findall(backtick_pattern, error_msg)
        if matches:
            # Get the longest match (likely the actual response)
            longest_match = max(matches, key=len)
            if len(longest_match) > 50:  # Reasonable response length
                return longest_match
        
        # Strategy 2: Look for "Could not parse LLM output:" and extract what follows
        parse_pattern = r"Could not parse LLM output:\s*(.+?)(?:\n\n|\Z)"
        match = re.search(parse_pattern, error_msg, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Strategy 3: Look for actual analysis content in the error
        # Sometimes the error contains the actual analysis wrapped in error text
        lines = error_msg.split('\n')
        content_lines = []
        capturing = False
        
        for line in lines:
            # Start capturing after common error prefixes
            if any(phrase in line for phrase in ["analysis", "data", "findings", "results"]):
                capturing = True
            
            if capturing and line.strip():
                # Skip obvious error message lines
                if not any(phrase in line.lower() for phrase in [
                    "error", "traceback", "exception", "could not parse", "parsing"
                ]):
                    content_lines.append(line)
        
        if content_lines:
            return '\n'.join(content_lines)
        
        return None
    
    def _direct_llm_fallback(self, prompt):
        """Direct LLM call as fallback when agent fails."""
        try:
            if not self.llm:
                return "I apologize, but I'm having technical difficulties processing your request."
            
            # Create a simplified prompt with data context
            df_info = f"Dataset shape: {self.df.shape}\nColumns: {', '.join(self.df.columns[:10])}"
            if len(self.df.columns) > 10:
                df_info += f"... and {len(self.df.columns) - 10} more columns"
            
            fallback_prompt = f"""
            As a data analyst, please analyze this request: "{prompt}"
            
            Dataset Information:
            {df_info}
            
            Please provide a comprehensive analysis with:
            1. Executive Summary
            2. Analysis approach
            3. Key insights based on the request
            4. Recommendations
            5. Next steps
            
            Note: I cannot execute code directly in this mode, so focus on analytical insights and methodology.
            """
            
            response = self.llm.invoke(fallback_prompt)
            
            # Add a note about the fallback mode
            fallback_response = f"""
## Analysis Report (Direct Mode)

*Note: This analysis was generated in direct mode due to technical constraints. For interactive visualizations and code execution, please try rephrasing your question.*

{response.content if hasattr(response, 'content') else str(response)}
            """
            
            return fallback_response
            
        except Exception as e:
            return f"""
## Analysis Error

I apologize, but I encountered technical difficulties processing your request. 

**Error Details:** {str(e)}

**Suggestions:**
1. Try rephrasing your question in simpler terms
2. Break complex requests into smaller parts
3. Ensure your CSV data is properly formatted
4. Check if the column names in your question match the dataset

**Available Columns:** {', '.join(self.df.columns[:5])}{'...' if len(self.df.columns) > 5 else ''}
            """
    
    def display_question_input(self):
        """Display the question input field with minimal styling"""
        return st.chat_input("Ask me anything about your data...")


class DataFrameUtils:
    """Utility class for dataframe operations"""
    
    @staticmethod
    def display_dataframe_info(df):
        """Display information about the dataframe."""
        st.markdown("<h3 style='text-align: center;'>Dataset Overview</h3>", unsafe_allow_html=True)
        # Show sample data in expander
        with st.expander("View sample data"):
            st.dataframe(df.head(), use_container_width=True)
        
class DataApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        self.df = None
        self.file_path = None
        self.response_processor = None
        self.analysis_agent = None
        
    def process_uploaded_file(self, file, google_api_key=None):
        """Process the uploaded CSV file and return dataframe and file path."""
        with st.spinner("Loading dataset..."):
            try:
                # Create temporary file
                with NamedTemporaryFile(delete=False) as f:
                    f.write(file.getbuffer())
                    self.file_path = f.name
                
                # Load dataframe with error handling
                try:
                    self.df = pd.read_csv(self.file_path)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    st.info("Make sure your file is a valid CSV with proper formatting.")
                    return None
                
                # Initialize components
                self.response_processor = ResponseProcessor(self.df)
                self.analysis_agent = DataAnalysisAgent(self.df, self.response_processor, google_api_key)
                
                return self.df
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return None
    
    def run(self):
        """Run the main application"""
        # Configure Streamlit page
        st.set_page_config(
            page_title="Analyzia - AI Data Analysis",
            page_icon="🤖", 
            layout="centered",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/ahammadnafiz/Analyzia',
                'Report a bug': 'https://github.com/ahammadnafiz/Analyzia/issues',
                'About': "Analyzia - AI-Powered Data Analysis Platform"
            }
        )
        
        # Hide Streamlit default elements for cleaner look while preserving functionality
        st.markdown("""
        <style>
        /* Hide Streamlit branding but keep essential elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Keep header visible but hide only the GitHub icon and deploy button */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
        
        /* Ensure sidebar remains visible and functional */
        .css-1d391kg {display: block !important;}
        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
        }
        
        /* Main content area */
        .main .block-container {
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 800px;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: transparent;
            border: none;
        }
        
        /* Chat input styling - control width */
        .stChatInputContainer {
            max-width: 800px;
            margin: 0 auto;
        }
        
        /* Fixed chat input container */
        div[data-testid="stChatInputContainer"] {
            max-width: 800px !important;
            margin: 0 auto !important;
            padding: 0 1rem;
        }
        
        /* Chat input field styling */
        div[data-testid="stChatInputContainer"] > div {
            max-width: 100% !important;
        }
        
        /* Ensure sidebar toggle button is visible */
        button[data-testid="collapsedControl"] {
            display: block !important;
            visibility: visible !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar with modern styling
        with st.sidebar:
            # Add custom CSS to ensure sidebar functionality
            st.markdown("""
            <style>
            /* Additional sidebar styling to ensure visibility */
            .css-1lcbmhc.e1fqkh3o0 {
                width: 250px !important;
                min-width: 250px !important;
            }
            
            /* Sidebar content styling */
            .css-17eq0hr {
                padding: 1rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown("# 🤖 Analyzia")
            st.markdown("*AI-Powered Data Analysis*")
            st.markdown("---")
            
            # API Key section
            st.markdown("### API Configuration")
            if GOOGLE_API_KEY:
                st.success("API Key configured")
                google_api_key = GOOGLE_API_KEY
            else:
                google_api_key = st.text_input("Google API Key", type="password", placeholder="Enter your API key...")
            
            st.markdown("---")
            
            # File upload section
            st.markdown("### Data Upload")
            uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
        
        # Initialize or reset session state if needed
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Process file if uploaded
        if uploaded_file and (self.df is None or uploaded_file.name != getattr(st.session_state, 'last_file', None)):
            self.process_uploaded_file(uploaded_file, google_api_key)
            st.session_state.last_file = uploaded_file.name if self.df is not None else None
            
            # Reset chat history when new file is uploaded
            st.session_state.messages = []
        
        # Setup agent if conditions are met
        if self.df is not None and google_api_key:
            if self.analysis_agent and self.analysis_agent.agent is None:
                # Update API key if changed
                self.analysis_agent.google_api_key = google_api_key
                self.analysis_agent.setup_agent(self.file_path)
        
        # Main content area - always show chat interface with title
        # Always show the Analyzia title at the top center
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="margin-bottom: 0.5rem; font-size: 3rem; font-weight: bold;">Analyzia</h1>
            <p style="color: #666; font-size: 1.1rem; margin: 0;">
                AI-Powered Data Analysis Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show dataset overview if dataset is uploaded
        if self.df is not None:
            DataFrameUtils.display_dataframe_info(self.df)
        
        # Display status information if setup is incomplete
        if not uploaded_file and not google_api_key:
            # Welcome screen when nothing is set up
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <p style="color: #0066cc; background-color: #e6f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0 auto; max-width: 600px;">
                👈 Get started by uploading a file and entering your API key in the sidebar
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif not uploaded_file:
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <p style="color: #0066cc; background-color: #e6f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0 auto; max-width: 600px;">
                    📁 Please upload a CSV file in the sidebar to get started.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif not google_api_key:
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <p style="color: #0066cc; background-color: #e6f3ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0 auto; max-width: 600px;">
                    🔑 Please enter your Google API key in the sidebar to start analyzing your data.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat messages container - always visible
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input - always visible at bottom
        if prompt := st.chat_input("Ask me anything about your data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response with validation
            with st.chat_message("assistant"):
                # Check if setup is complete
                if not uploaded_file:
                    response = """
🚫 **No Dataset Found**

I'd love to help you analyze your data, but I don't see any dataset uploaded yet.

**To get started:**
1. 📁 Upload a CSV file using the sidebar
2. 🔑 Enter your Google API key in the sidebar
3. 🚀 Ask your question again

Once you've uploaded your data, I can help you with:
- Data exploration and summaries
- Statistical analysis and correlations  
- Beautiful visualizations and charts
- Business insights and recommendations
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                elif not google_api_key:
                    response = """
🔑 **API Key Required**

I can see your dataset, but I need a Google API key to analyze it for you.

**To continue:**
1. 🔑 Enter your Google API key in the sidebar (you can get one from Google AI Studio)
2. 🚀 Ask your question again

Your data is ready - I just need the API key to start the analysis!
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                elif self.analysis_agent and self.analysis_agent.agent:
                    # Everything is set up - proceed with analysis
                    response = self.analysis_agent.handle_chat_input(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                else:
                    response = """
⚠️ **Setup Issue**

There seems to be an issue with the analysis setup. Please try:

1. 🔄 Refresh the page
2. 📁 Re-upload your CSV file
3. 🔑 Re-enter your API key
4. 🚀 Ask your question again

If the problem persists, please check that your API key is valid and your CSV file is properly formatted.
                    """
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


# Run the application
if __name__ == "__main__":
    app = DataApp()
    app.run()