library(dplyr)
library(ggplot2)
library(mediation)
library(ggpubr) # 시각화용

# 1. 데이터 로드 및 전처리
# 경로는 본인 환경에 맞게 유지
df <- read.csv("./dataset.csv")

# Factor 변환
df$Group <- factor(df$Group, levels = c("NC", "CDM"))
df$Sex <- factor(df$Sex)

# 변수 리스트 설정
species_list <- c("s__Veillonella_parvula", "s__Lactobacillus_crispatus")
gene_list <- c("K00075", "K01512", "K03060", "K03101", "K03116", "K03687", "K03799", "K04042", "K10716")
outcome <- "MDF"
covariates <- c("Age", "Sex", "BMI", "Group")

# 결과 저장용 리스트
results_list <- list()

set.seed(42) # 재현성 확보

print("Starting Mediation Analysis...")

# 2. Mediation Loop
count <- 1
for (sp in species_list) {
  for (gene in gene_list) {
    
    tryCatch({
      # --- Step 1: Mediator Model (Gene ~ Species + Covariates) ---
      # Family를 gaussian으로 명시 (연속형 변수일 경우)
      f_m <- as.formula(paste(gene, "~", sp, "+", paste(covariates, collapse="+")))
      mod_m <- lm(f_m, data = df)
      
      # --- Step 2: Outcome Model (MDF ~ Species + Gene + Covariates) ---
      f_y <- as.formula(paste(outcome, "~", sp, "+", gene, "+", paste(covariates, collapse="+")))
      mod_y <- lm(f_y, data = df)
      
      # --- Step 3: Run Mediation ---
      # sims=1000 권장 (부트스트랩 횟수)
      med_out <- mediate(mod_m, mod_y, treat = sp, mediator = gene, boot = TRUE, sims = 1000)
      
      # 결과 추출 (ACME: Average Causal Mediation Effect)
      results_list[[count]] <- data.frame(
        Species = gsub("s__", "", sp),  # 이름 정리
        Gene = gene,
        ACME_Estimate = med_out$d0,      # ACME (간접 효과)
        ACME_Lower = med_out$d0.ci[1],   # 95% CI Lower
        ACME_Upper = med_out$d0.ci[2],   # 95% CI Upper
        ACME_P = med_out$d0.p,           # P-value
        ADE_Estimate = med_out$z0,       # ADE (직접 효과)
        Total_Estimate = med_out$tau.coef, # 총 효과
        Prop_Mediated = med_out$n0       # 매개 비율 (Proportion Mediated)
      )
      count <- count + 1
      
    }, error = function(e) {
      message(paste("Error processing:", sp, "->", gene, ":", e$message))
    })
  }
}

# 데이터프레임 병합
med_df <- do.call(rbind, results_list)

# ==============================================================================
# 3. 통계 보정 및 필터링 (중요!)
# ==============================================================================

# 1) FDR (Benjamini-Hochberg) 보정 추가
# 전체 테스트 셋에 대해 P-value 보정을 수행합니다.
med_df$ACME_FDR <- p.adjust(med_df$ACME_P, method = "fdr")

# 2) 유의성 그룹 지정 (FDR < 0.1 또는 P < 0.05 등 기준 설정)
# 여기서는 P-value < 0.05를 기준으로 하되, FDR도 확인 가능하게 남겨둡니다.
med_df$Significance <- ifelse(med_df$ACME_P < 0.05, "Significant", "Ns")

# 3) 라벨 생성 (유의한 경우에만 매개 비율 표시)
med_df$Label <- ifelse(med_df$Significance == "Significant", 
                       paste0(sprintf("%.1f", med_df$Prop_Mediated * 100), "%"), 
                       "")

# ==============================================================================
# 4. 시각화 (Publication Quality)
# ==============================================================================

# 유의한 순서대로 정렬 (Plot팅을 예쁘게 하기 위함)
med_df <- med_df %>% 
  group_by(Species) %>% 
  arrange(desc(ACME_P)) %>% 
  mutate(Gene = factor(Gene, levels = unique(Gene)))

