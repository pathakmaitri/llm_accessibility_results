research analysis:


=======================================================
1. MODEL PERFORMANCE COMPARISON
=======================================================

Mean Total Score per Model:
        Mean Total Score  Std Dev    N
Model                                 
Claude             12.00     1.93  120
GPT                11.58     2.01  120
Gemini             10.97     2.41  120

Mean Score per Metric per Model:
        False_Positives  Detection_Accuracy  Implementation_Accuracy  Code_Reasoning  Violation_Presence
Model                                                                                                   
Claude             1.02                2.88                     2.70            2.67                2.76
GPT                1.17                2.84                     2.62            2.28                2.62
Gemini             1.27                2.77                     2.34            2.22                2.37

One-Way ANOVA: F = 7.152, p = 0.0009
   Statistically significant difference between models (p < 0.05)

=======================================================
2. PROMPT COMPARISON
=======================================================

Mean Total Score per Prompt:
        Mean Total Score  Std Dev    N
Prompt                                
P1                 11.20     2.35  180
P2                 11.83     1.92  180

P2 vs P1 % Change: +5.6%

Paired t-test: t = -4.055, p = 0.0001
   Statistically significant difference between prompts (p < 0.05)

=======================================================
3. SUCCESS RATE  (Total Score = 15 = perfect)
=======================================================

Success Rate per Model:
        Successes  Total Runs  Success %
Model                                   
Claude         17         120       14.2
GPT            11         120        9.2
Gemini          7         120        5.8

Success Rate per Prompt:
        Successes  Total Runs  Success %
Prompt                                  
P1             20         180       11.1
P2             15         180        8.3

=======================================================
4. ERROR ANALYSIS  (Score 0 or 1 = failure)
=======================================================

Failure Count per Metric per Model (score ≤ 1):
        False\nPositives  Detection\nAccuracy  Implementation\nAccuracy  Code\nReasoning  Violation\nPresence
Model                                                                                                        
Claude                74                    0                         2                2                    0
GPT                   70                    0                         4                9                    3
Gemini                72                    1                         8               10                    6

=======================================================
5. ISSUE-TYPE BREAKDOWN
=======================================================

Mean Total Score by Issue Type and Model:
Model                     Claude   GPT  Gemini
Issue_Type                                    
Aria Labels                11.30  10.5    8.40
Color Contrast             13.35  13.3   13.40
Empty Buttons              12.10  11.5    9.45
Empty Links                 9.50   9.6    9.95
Missing Form Input Label   11.85  11.6   11.10
Non-text Context           13.90  13.0   13.50

=======================================================
6. COMPARISON OF VIOLATIONS
=======================================================

Mean Total Score per Violation Type:
                          Mean Total Score  Std Dev   N
Issue_Type                                             
Aria Labels                          10.07     1.66  60
Color Contrast                       13.35     1.61  60
Empty Buttons                        11.02     1.84  60
Empty Links                           9.68     1.51  60
Missing Form Input Label             11.52     1.31  60
Non-text Context                     13.47     1.65  60

   Found 6 issue types: ['Color Contrast', 'Non-text Context', 'Missing Form Input Label', 'Empty Buttons', 'Aria Labels', 'Empty Links']
   Independent t-test requires exactly 2 groups - skipping.