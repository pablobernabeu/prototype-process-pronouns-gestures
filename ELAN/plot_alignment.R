
library(ggplot2)
library(dplyr)
library(ggrepel)

# Read the matched results file
df <- read_csv("preprocessed_data/alignment_data.csv")

# Convert match_index to a factor for categorical plotting
df$match_index <- as.factor(df$match_index)

# Separate demonstrative pronouns and gesture apex data
pronouns <- df %>% filter(modality == "demonstrative_pronoun")
gestures <- df %>% filter(modality == "gesture_apex")

# Match gesture apex to closest demonstrative pronoun within 2000ms
matches <- gestures %>%
  rowwise() %>%
  mutate(
    closest_pronoun = pronouns$onset[which.min(abs(pronouns$onset - onset))],
    alignment_difference = onset - closest_pronoun
  ) %>%
  filter(abs(alignment_difference) <= 2000)

# Determine dynamic positions
x_min <- min(matches$alignment_difference, na.rm = TRUE)
x_max <- max(matches$alignment_difference, na.rm = TRUE)
y_max_scatter <- nrow(matches) + 1  # Slightly above the highest point
y_max_histogram <- max(table(cut(matches$alignment_difference, breaks = 10))) * 0.9  # 90% of max frequency

# Scatter plot: Temporal alignment of gesture-pronoun pairs
(
  ggplot(df, aes(x = alignment_difference, y = match_index)) +
    geom_point(size = 3, color = "blue") +
    geom_text_repel(aes(label = demonstrative_pronoun), hjust = 0, vjust = 1, size = 4) +
    geom_vline(xintercept = 0, color = "darkgrey", linetype = "dashed") +
    labs(title = "Temporal Alignment of Gestures and Pronouns",
         x = "Time Difference (Apex Onset - Pronoun Onset) in ms",
         y = "Pronoun-Gesture Pair") +
    theme_minimal(base_size = 16) +
    annotate("text", x = x_min * 0.5, y = y_max_scatter * 0.85, label = "Gesture apex before\ndemonstrative pronoun",
             colour = "darkgrey", hjust = 0.7, size = 4) +
    annotate("text", x = x_max * 0.5, y = y_max_scatter * 0.85, label = "Demonstrative pronoun\nbefore gesture apex",
             colour = "darkgrey", hjust = 0.3, size = 4)
) %>%
  ggsave(filename = 'scatter_plot.png', path = 'plots', 
         width = 9, height = 5)

# Histogram: Distribution of alignment differences
(
  ggplot(df, aes(x = alignment_difference)) +
    geom_histogram(bins = 20, fill = "blue", alpha = 0.7, color = "black") +
    geom_vline(xintercept = 0, color = "darkgrey", linetype = "dashed") +
    labs(title = "Distribution of Alignment Differences",
         x = "Time Difference (Apex Onset - Pronoun Onset) in ms",
         y = "Frequency") +
    theme_minimal() +
    annotate("text", x = -1000, y = max(table(cut(matches$alignment_difference, breaks = 10))) * 0.9, 
             label = "Gesture apex before demonstrative pronoun", colour = "darkgrey", hjust = 1, size = 5) +
    annotate("text", x = 1000, y = max(table(cut(matches$alignment_difference, breaks = 10))) * 0.9, 
             label = "Demonstrative pronoun before gesture apex", colour = "darkgrey", hjust = 0, size = 5)
) %>%
  ggsave(filename = 'histogram.png', path = 'plots')
