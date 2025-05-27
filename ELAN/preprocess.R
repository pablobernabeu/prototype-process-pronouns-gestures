
library(dplyr)
library(readr)

# Read CSV file, skipping the second column
df <- read_csv("ELAN_export/1.mp4_PabloBernabeu.csv", col_names = FALSE) %>%
  select(-X2)  # Remove second column

# Rename columns
colnames(df) <- c("modality", "onset", "offset", "duration", "demonstrative_pronoun")

# Separate demonstrative pronouns and gesture apices
pronouns <- df %>% filter(modality == "demonstrative_pronoun")
apices <- df %>% filter(modality == "gesture_apex")

# Sort by onset time
pronouns <- pronouns %>% arrange(onset)
apices <- apices %>% arrange(onset)

# Initialize match_index column
pronouns$match_index <- NA
apices$match_index <- NA

match_counter <- 1  # Start match index counter

for (i in seq_len(nrow(pronouns))) {
  pronoun_onset <- pronouns$onset[i]
  
  # Find the closest gesture apex within 2000 ms
  potential_matches <- apices %>%
    filter(is.na(match_index)) %>%
    filter(abs(onset - pronoun_onset) <= 2000) %>%
    arrange(abs(onset - pronoun_onset))  # Sort by proximity
  
  if (nrow(potential_matches) > 0) {
    best_match_index <- potential_matches$onset[1]  # Take the closest match
    
    # Assign match_index
    pronouns$match_index[i] <- match_counter
    apices$match_index[apices$onset == best_match_index] <- match_counter
    
    match_counter <- match_counter + 1  # Increment match counter
  }
}

# Combine results 
df_matched <- bind_rows(pronouns, apices) %>% arrange(onset)

# Compute temporal difference (gesture onset - pronoun onset)
df_matched <- df_matched %>%
  filter(!is.na(match_index)) %>%  # Only consider matched pairs
  group_by(match_index) %>%
  mutate(alignment_difference = onset[modality == "gesture_apex"] - onset[modality == "demonstrative_pronoun"]) %>%
  ungroup()

# Ensure alignment_difference is numeric
df_matched$alignment_difference <- as.numeric(df_matched$alignment_difference)

# Export data
write_csv(df_matched, "preprocessed_data/1_alignment_data.csv", )

# Print results
print(df_matched)
