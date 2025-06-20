package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Конфигурация приложения
type Config struct {
	AudioAutoThreshold        bool     `json:"audio_auto_threshold"`
	MotionAutoThreshold       bool     `json:"motion_auto_threshold"`
	AudioThreshold            float64  `json:"audio_threshold"`
	MotionThreshold           float64  `json:"motion_threshold"`
	SpeechThresholdMultiplier float64  `json:"speech_threshold_multiplier"` // 5.0
	MinSpeechDuration         float64  `json:"min_speech_duration"`         // 0.3
	MinSilenceDuration        float64  `json:"min_silence_duration"`
	MaxSpeechGap              float64  `json:"max_speech_gap"`
	SpeechDetection           bool     `json:"speech_detection"`
	NoiseFloor                float64  `json:"noise_floor"` // Уровень фонового шума
	GameEventsEnabled         bool     `json:"game_events_enabled"`
	HighlightPadding          float64  `json:"highlight_padding"`
	Codec_Args                []string `json:"Codec_Args"`          // Дополнительные аргументы кодека
	GPUAcceleration           []string `json:"gpu_acceleration"`    // Использовать GPU для видео обработки
	OutputDir                 string   `json:"output_dir"`
	TempDir                   string   `json:"temp_dir"`
	PreviewGeneration         bool     `json:"preview_generation"`
	TransitionDuration        float64  `json:"transition_duration"`
}

// Структура для хранения сегментов видео
type ClipSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Type    string  `json:"type"`    // audio, motion, face, emotion, game
	Score   float64 `json:"score"`   // Оценка важности момента (0-1)
	Emotion string  `json:"emotion"` // Для эмоциональных событий
	Details string  `json:"details"` // Дополнительная информация
	Info    string  `json:"info"`    // Дополнительная информация
}

// Структура для игровых событий
type GameEvent struct {
	Timestamp float64 `json:"timestamp"`
	EventType string  `json:"event_type"`
	Details   string  `json:"details"`
	Intensity float64 `json:"intensity"`
}

func main() {
	// Обработка аргументов командной строки
	configPath := flag.String("config", "config.json", "Path to config file")
	inputVideo := flag.String("input", "/run/media/sasha/video/2024-01-15 19-07-24.mp4", "Input video file")
	outputName := flag.String("output", "", "Output video name (without extension)")
	flag.Parse()

	if *inputVideo == "" {
		fmt.Println("Please specify input video with -input flag")
		os.Exit(1)
	}

	// Загрузка конфигурации
	config, err := loadConfig(*configPath)
	if err != nil {
		log.Fatalf("Error loading config: %v", err)
	}

	// Создание временной директории
	if err := os.MkdirAll(config.TempDir, 0750); err != nil {
		log.Fatalf("Error creating temp dir: %v", err)
	}

	// Создание выходной директории
	if err := os.MkdirAll(config.OutputDir, 0750); err != nil {
		log.Fatalf("Error creating output dir: %v", err)
	}

	// Генерация имени выходного файла
	timestamp := time.Now().Format("20060102-150405")
	if *outputName == "" {
		*outputName = fmt.Sprintf("highlight_%s", timestamp)
	}
	outputVideo := filepath.Join(config.OutputDir, *outputName+".mp4")
	previewVideo := filepath.Join(config.OutputDir, *outputName+"_preview.mp4")

	fmt.Println("Starting video analysis for:", *inputVideo)
	fmt.Printf("Config: %+v\n", config)

	// Параллельный анализ

	var audioEvents []ClipSegment
	var videoEvents []ClipSegment
	var gameEvents []ClipSegment

	startTime := time.Now()

	if config.AudioAutoThreshold {
		config.AudioThreshold = -999 // Флаг для автонастройки
	}

	if config.MotionAutoThreshold {
		config.MotionThreshold = -1 // Флаг для автонастройки
	}

	// Анализ аудио
	func() {
		audioEvents = detectAudioEvents(*inputVideo, config)

		// Дополнительная детекция речи
		if config.SpeechDetection {
			speechEvents := detectSpeechActivity(*inputVideo, config)
			audioEvents = combineAudioEvents(audioEvents, speechEvents)
		}

		fmt.Printf("Detected %d audio events\n", len(audioEvents))
	}()

	// Анализ видео
	func() {
		videoEvents = detectVideoEvents(*inputVideo, config)
		fmt.Printf("\nDetected %d video events\n", len(videoEvents))
	}()

	// Анализ игровых событий
	func() {
		gameEvents = detectGameEvents(*inputVideo, config)
		fmt.Printf("\nDetected %d game events\n", len(gameEvents))
	}()

	analysisDuration := time.Since(startTime)
	fmt.Printf("Analysis completed in %s\n", analysisDuration.Round(time.Second))

	// Комбинирование и оптимизация сегментов
	fmt.Println("Combining events...")
	allSegments := append(audioEvents, append(videoEvents, gameEvents...)...)
	segments := combineEvents(allSegments, config)
	segments = combineEvents_mergeted(segments, config)
	fmt.Printf("Created %d highlight segments\n", len(segments))

	// Сохранение информации о сегментах
	segmentsFile := filepath.Join(config.OutputDir, *outputName+"_segments.json")
	if err := saveSegments(segments, segmentsFile); err != nil {
		log.Printf("Warning: failed to save segments info: %v", err)
	}

	// Рендеринг финального видео
	fmt.Println("Rendering final video...")
	renderStart := time.Now()

	if err := renderFinalVideo(*inputVideo, segments, outputVideo, config); err != nil {
		log.Fatalf("Rendering failed: %v", err)
	}

	renderDuration := time.Since(renderStart)
	fmt.Printf("\nRendering completed in %s\n", renderDuration.Round(time.Second))

	// Генерация превью
	if config.PreviewGeneration {
		fmt.Println("Generating preview video...")
		if err := generatePreview(outputVideo, previewVideo); err != nil {
			log.Printf("Preview generation failed: %v", err)
		} else {
			fmt.Println("Preview generated:", previewVideo)
		}
	}

	fmt.Println("Done! Output file:", outputVideo)
	fmt.Println("Segments info:", segmentsFile)
}

// Загрузка конфигурации
func loadConfig(path string) (Config, error) {
	config := Config{
		AudioAutoThreshold:        true,
		MotionAutoThreshold:       true,
		AudioThreshold:            -40.0,
		MotionThreshold:           0.04,
		SpeechThresholdMultiplier: 5.0,
		MinSpeechDuration:         0.3,
		MinSilenceDuration:        0.3,
		MaxSpeechGap:              0.5, // Объединять паузы короче 500ms
		SpeechDetection:           true,
		NoiseFloor:                -60.0,
		GameEventsEnabled:         false,
		HighlightPadding:          1.0,
		Codec_Args:                []string{},
		GPUAcceleration:           []string{},
		OutputDir:                 "output",
		TempDir:                   "temp",
		PreviewGeneration:         false,
		TransitionDuration:        0.5,
	}

	cleanPath := filepath.Clean(path)
	if _, err := os.Stat(cleanPath); err == nil {
		data, err := os.ReadFile(cleanPath)
		if err != nil {
			return config, fmt.Errorf("error reading config file: %w", err)
		}

		if err := json.Unmarshal(data, &config); err != nil {
			return config, fmt.Errorf("error parsing config file: %w", err)
		}
	}

	return config, nil
}

func getAudioAnalysisCachePath(videoPath string) string {
	// Генерируем уникальное имя файла на основе пути к видео
	hash := sha256.Sum256([]byte(videoPath))
	return filepath.Join(os.TempDir(), fmt.Sprintf("audio_analysis_%x.txt", hash))
}

// Детекция аудио событий (пики громкости)
func detectAudioEvents(videoPath string, config Config) []ClipSegment {
	fmt.Print("Analyzing audio: ")
	cachePath := getAudioAnalysisCachePath(videoPath)
	var scanner *bufio.Scanner
	var cmd *exec.Cmd

	// Проверяем наличие кэшированных данных
	cleanCachePath := filepath.Clean(cachePath)
	if _, err := os.Stat(cleanCachePath); os.IsNotExist(err) {
		// Создаем команду FFmpeg с анализом частот речи
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd = exec.CommandContext(ctx, // #nosec G204
			"ffmpeg",
			"-i", videoPath,
			"-filter_complex", "astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
			"-f", "null", "-",
		)

		stderr, err := cmd.StderrPipe()
		if err != nil {
			log.Printf("Audio detection error: %v", err)
			return nil
		}

		cacheFile, err := os.Create(cleanCachePath)
		if err != nil {
			log.Printf("Cache creation error: %v", err)
		}
		defer func() {
			if err := cacheFile.Close(); err != nil {
				log.Printf("Error closing cache file: %v", err)
			}
		}()

		if err := cmd.Start(); err != nil {
			log.Printf("Audio detection error: %v", err)
			return nil
		}

		tee := io.TeeReader(stderr, cacheFile)
		scanner = bufio.NewScanner(tee)
	} else {
		fmt.Print("Using cached data")
		file, err := os.Open(cleanCachePath)
		if err != nil {
			log.Printf("Cache open error: %v", err)
			return nil
		}
		defer func() {
			if err := file.Close(); err != nil {
				log.Printf("Error closing cache file: %v", err)
			}
		}()
		scanner = bufio.NewScanner(file)
	}

	// Обработка данных
	reTime := regexp.MustCompile(`pts_time:(\d+(\.\d+)?)`)
	reRMS := regexp.MustCompile(`lavfi\.astats\.Overall\.RMS_level=(-?\d+\.\d+)`)

	type audioPoint struct {
		time float64
		rms  float64
	}
	var points []audioPoint
	var rmsValues []float64
	var currentTime float64 = -1

	for scanner.Scan() {
		line := scanner.Text()

		if matches := reTime.FindStringSubmatch(line); matches != nil {
			if time, err := strconv.ParseFloat(matches[1], 64); err == nil {
				currentTime = time
			}
		}

		if matches := reRMS.FindStringSubmatch(line); matches != nil {
			if rms, err := strconv.ParseFloat(matches[1], 64); err == nil {
				if rms > -100 {
					rmsValues = append(rmsValues, rms)
					points = append(points, audioPoint{time: currentTime, rms: rms})
				}
			}
		}
	}

	// Автоматический подбор порога
	baseThreshold := config.AudioThreshold
	var dynamicThreshold float64
	noiseFloor := config.NoiseFloor // Минимальный уровень шума

	if config.AudioAutoThreshold && len(rmsValues) > 0 {
		sort.Float64s(rmsValues)
		idx := int(float64(len(rmsValues)) * 0.95)
		if idx >= len(rmsValues) {
			idx = len(rmsValues) - 1
		}
		baseThreshold = rmsValues[idx] * 0.95
		dynamicThreshold = baseThreshold + 3.0

		if dynamicThreshold < noiseFloor+6 {
			dynamicThreshold = noiseFloor + 6
		}
		fmt.Printf("\r[Audio] Auto threshold: base=%.2fdB dynamic=%.2fdB\n",
			baseThreshold, dynamicThreshold)
	} else {
		dynamicThreshold = baseThreshold
	}

	// Детекция событий с гистерезисом
	events := []ClipSegment{}
	var eventStart float64 = -1
	var eventEndConfirmation float64 = -1
	minEventDuration := 0.3
	minSilenceDuration := config.MinSilenceDuration
	if minSilenceDuration <= 0 {
		minSilenceDuration = 0.2 // Default value
	}

	for _, p := range points {
		if p.rms >= dynamicThreshold {
			// Сброс подтверждения окончания при новом звуке
			eventEndConfirmation = -1

			// Начало нового события
			if eventStart < 0 {
				eventStart = p.time
			}
		} else if eventStart >= 0 {
			// Начали отсчет подтверждения окончания
			if eventEndConfirmation < 0 {
				eventEndConfirmation = p.time
			}

			// Проверяем, прошло ли достаточно времени тишины
			if p.time-eventEndConfirmation >= minSilenceDuration {
				duration := eventEndConfirmation - eventStart
				if duration >= minEventDuration {
					events = append(events, ClipSegment{
						Start: eventStart - config.HighlightPadding,
						End:   eventEndConfirmation + config.HighlightPadding,
						Type:  "audio",
						Score: 0.7 + (duration * 0.1),
					})
				}
				eventStart = -1
				eventEndConfirmation = -1
			}
		}
	}

	// Обработка последнего события
	if eventStart >= 0 {
		endTime := points[len(points)-1].time
		if eventEndConfirmation > 0 {
			endTime = eventEndConfirmation
		}
		events = append(events, ClipSegment{
			Start: eventStart - config.HighlightPadding,
			End:   endTime + config.HighlightPadding,
			Type:  "audio",
			Score: 0.8,
		})
	}

	// Объединение близких событий
	events = mergeCloseAudioEvents(events, config.MaxSpeechGap)

	if cmd != nil {
		if err := cmd.Wait(); err != nil {
			log.Printf("Audio detection finished with error: %v", err)
			if err := os.Remove(cleanCachePath); err != nil {
				log.Printf("Failed to remove cache file: %v", err)
			}
		}
	}

	fmt.Printf("\rAnalyzing audio: [%d events detected]%s\n", len(events), strings.Repeat(" ", 30))
	return events
}

// Вспомогательная функция для объединения близких аудио-событий
func mergeCloseAudioEvents(events []ClipSegment, maxGap float64) []ClipSegment {
	if len(events) == 0 || maxGap <= 0 {
		return events
	}

	// Сортировка по времени начала
	sort.Slice(events, func(i, j int) bool {
		return events[i].Start < events[j].Start
	})

	merged := []ClipSegment{events[0]}

	for i := 1; i < len(events); i++ {
		last := &merged[len(merged)-1]
		current := events[i]

		// Рассчитываем интервал между событиями
		gap := current.Start - last.End

		// Если интервал меньше допустимого - объединяем
		if gap <= maxGap {
			last.End = current.End
			last.Score = (last.Score + current.Score) / 2
			last.Details = "Merged audio event"
		} else {
			merged = append(merged, current)
		}
	}

	return merged
}

// Детекция видео событий (движение и лица) с кэшированием
func detectVideoEvents(videoPath string, config Config) []ClipSegment {
	fmt.Println("Detecting video events...")
	// Генерируем уникальный хеш для конфигурации
	configHash := fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf(
		"%.4f|%.4f",
		config.MotionThreshold,
		config.HighlightPadding,
	))))

	// Формируем имя кэш-файла
	cacheFileName := filepath.Clean(fmt.Sprintf("%s.video_events.%s.json", videoPath, configHash[:8]))

	// Пытаемся прочитать данные из кэша
	if cacheData, err := os.ReadFile(cacheFileName); err == nil {
		var cachedEvents []ClipSegment
		if err := json.Unmarshal(cacheData, &cachedEvents); err == nil {
			log.Printf("Loaded cached video events: %d segments from %s",
				len(cachedEvents), cacheFileName)
			return cachedEvents
		} else {
			log.Printf("Video cache parse error: %v - regenerating...", err)
		}
	} else {
		log.Printf("Video cache not found: %s - analyzing video...", cacheFileName)
	}

	events := []ClipSegment{}

	// Детекция движения
	fmt.Print("  Motion detection: ")
	motionEvents := detectMotionFFmpeg(videoPath, config)
	fmt.Print("\r  Motion detection: complete\n")
	for _, t := range motionEvents {
		events = append(events, ClipSegment{
			Start: maxFloat(0, t-config.HighlightPadding),
			End:   t + config.HighlightPadding,
			Type:  "motion",
			Score: 0.7,
		})
	}

	// Сохраняем результаты в кэш
	if jsonData, err := json.MarshalIndent(events, "", "  "); err == nil {
		if err := os.WriteFile(cacheFileName, jsonData, 0600); err == nil {
			log.Printf("Saved %d video events to cache: %s",
				len(events), cacheFileName)
		} else {
			log.Printf("Failed to write video cache file: %v", err)
		}
	} else {
		log.Printf("Failed to marshal video cache data: %v", err)
	}

	return events
}

// Детекция движения через FFmpeg с кэшированием
func detectMotionFFmpeg(videoPath string, config Config) []float64 {
	autoThreshold := config.MotionThreshold < 0
	var threshold float64
	var sceneValues []float64

	// Сбор данных для автоматического определения порога
	if autoThreshold {
		fmt.Print("Calculating motion threshold...")
		cmd := exec.Command( // #nosec G204
			"ffmpeg",
			"-i", videoPath,
			"-vf", "select='gt(scene,0)',metadata=print",
			"-f", "null", "-",
		)

		stderr, err := cmd.StderrPipe()
		if err != nil {
			log.Printf("Motion stats error: %v", err)
			return nil
		}

		if err := cmd.Start(); err != nil {
			log.Printf("Motion stats error: %v", err)
			return nil
		}

		scanner := bufio.NewScanner(stderr)
		reScene := regexp.MustCompile(`scene:(\d+\.\d+)`)

		for scanner.Scan() {
			line := scanner.Text()
			if matches := reScene.FindStringSubmatch(line); matches != nil {
				if val, err := strconv.ParseFloat(matches[1], 64); err == nil {
					sceneValues = append(sceneValues, val)
				}
			}
		}

		if err := cmd.Wait(); err != nil {
			log.Printf("Motion stats finished with error: %v", err)
		}

		if len(sceneValues) > 0 {
			sort.Float64s(sceneValues)
			idx := int(float64(len(sceneValues)) * 0.9)
			if idx >= len(sceneValues) {
				idx = len(sceneValues) - 1
			}
			threshold = sceneValues[idx] * 1.1
			if threshold > 1.0 {
				threshold = 1.0
			}
			fmt.Printf("\rCalculating motion threshold: %.4f (based on %d samples)\n",
				threshold, len(sceneValues))
		} else {
			threshold = 0.1
			log.Println("No motion data found, using default threshold")
		}
	} else {
		threshold = config.MotionThreshold
	}

	// Генерация уникального хеша для кэша
	configHash := fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf(
		"%.4f|%t",
		threshold,
		autoThreshold,
	))))
	cacheFileName := filepath.Clean(fmt.Sprintf("%s.motion_events.%s.json", videoPath, configHash[:8]))

	// Проверка кэша
	if cacheData, err := os.ReadFile(cacheFileName); err == nil {
		var cachedEvents []float64
		if err := json.Unmarshal(cacheData, &cachedEvents); err == nil {
			log.Printf("Loaded cached motion events: %d points", len(cachedEvents))
			return removeDuplicates(cachedEvents)
		}
	}

	// Детекция движения
	fmt.Print("  Detecting motion events: ")
	cmd := exec.Command( // #nosec G204
		"ffmpeg",
		"-i", videoPath,
		"-vf", fmt.Sprintf("select='gt(scene\\,%f)',metadata=print", threshold),
		"-f", "null", "-",
	)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Printf("Motion detection error: %v", err)
		return nil
	}

	if err := cmd.Start(); err != nil {
		log.Printf("Motion detection error: %v", err)
		return nil
	}

	scanner := bufio.NewScanner(stderr)
	events := []float64{}
	reTime := regexp.MustCompile(`pts_time:(\d+\.\d+)`)

	for scanner.Scan() {
		line := scanner.Text()
		if matches := reTime.FindStringSubmatch(line); matches != nil {
			if time, err := strconv.ParseFloat(matches[1], 64); err == nil {
				events = append(events, time)
			}
		}
	}

	if err := cmd.Wait(); err != nil {
		log.Printf("Motion detection finished with error: %v", err)
	}

	// Удаление дубликатов и группировка событий
	uniqueEvents := removeDuplicates(events)
	groupedEvents := groupCloseEvents(uniqueEvents, 0.5) // Группируем события ближе 0.5s

	// Сохранение в кэш
	if jsonData, err := json.MarshalIndent(groupedEvents, "", "  "); err == nil {
		if writeErr := os.WriteFile(cacheFileName, jsonData, 0600); writeErr != nil {
			log.Printf("Failed to save motion events cache: %v", writeErr)
		}
	}

	fmt.Printf("\r  Detecting motion events: [%d events detected]%s\n", len(groupedEvents), strings.Repeat(" ", 30))
	return groupedEvents
}

// Группировка близких событий движения
func groupCloseEvents(events []float64, maxGap float64) []float64 {
	if len(events) == 0 {
		return events
	}

	sort.Float64s(events)
	grouped := []float64{events[0]}
	currentGroupStart := events[0]
	currentGroupEnd := events[0]

	for i := 1; i < len(events); i++ {
		if events[i]-currentGroupEnd <= maxGap {
			currentGroupEnd = events[i]
		} else {
			// Добавляем центр группы как представителя
			groupCenter := (currentGroupStart + currentGroupEnd) / 2
			grouped = append(grouped, groupCenter)

			currentGroupStart = events[i]
			currentGroupEnd = events[i]
		}
	}

	// Добавляем последнюю группу
	groupCenter := (currentGroupStart + currentGroupEnd) / 2
	grouped = append(grouped, groupCenter)

	return grouped
}

// Вспомогательные функции для float64
func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// Генерация безопасных случайных чисел
func secureRandIntn(n int) int {
	num, err := rand.Int(rand.Reader, big.NewInt(int64(n)))
	if err != nil {
		log.Printf("Secure random error: %v", err)
		return 0
	}
	return int(num.Int64())
}

func secureRandFloat() float64 {
	num, err := rand.Int(rand.Reader, big.NewInt(1<<53))
	if err != nil {
		log.Printf("Secure random error: %v", err)
		return 0.5
	}
	return float64(num.Int64()) / (1 << 53)
}

// Детекция игровых событий (имитация)
func detectGameEvents(videoPath string, config Config) []ClipSegment {
	if !config.GameEventsEnabled {
		return nil
	}

	fmt.Print("Detecting game events: ")
	// В реальной системе здесь будет интеграция с API игры
	// Для примера генерируем случайные события
	events := []ClipSegment{}

	// Получаем длительность видео
	duration, err := getVideoDuration(videoPath)
	if err != nil {
		log.Printf("Error getting video duration: %v", err)
		return nil
	}

	// Типы игровых событий
	eventTypes := []string{"kill", "death", "victory", "defeat", "achievement"}

	// Генерируем 5-10 случайных событий
	numEvents := 5 + secureRandIntn(6)
	for i := 0; i < numEvents; i++ {
		eventTime := secureRandFloat() * duration
		eventType := eventTypes[secureRandIntn(len(eventTypes))]
		intensity := 0.5 + secureRandFloat()*0.5 // 0.5-1.0

		events = append(events, ClipSegment{
			Start:   eventTime - config.HighlightPadding,
			End:     eventTime + config.HighlightPadding,
			Type:    "game",
			Score:   intensity,
			Details: fmt.Sprintf("Game event: %s", eventType),
		})

	}

	return events
}

// Получение длительности видео
func getVideoDuration(videoPath string) (float64, error) {
	cmd := exec.Command( // #nosec G204
		"ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath,
	)

	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}

	duration, err := strconv.ParseFloat(strings.TrimSpace(string(output)), 64)
	if err != nil {
		return 0, err
	}

	return duration, nil
}

// Удаление дубликатов событий
func removeDuplicates(times []float64) []float64 {
	seen := make(map[float64]bool)
	result := []float64{}
	for _, t := range times {
		if !seen[t] {
			seen[t] = true
			result = append(result, t)
		}
	}
	return result
}

// Комбинирование событий в сегменты видео
func combineEvents(segments []ClipSegment, config Config) []ClipSegment {
	if len(segments) == 0 {
		return nil
	}

	// Сортируем по времени начала
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Start < segments[j].Start
	})

	// Объединяем пересекающиеся сегменты
	// merged := combineEvents_mergeted(segments, config)

	merged := combineEvents_mergeted(segments, config)

	for i := range merged {
		// Применяем padding
		merged[i].Start -= config.HighlightPadding
		if merged[i].Start < 0 {
			merged[i].Start = 0
		}
		merged[i].End += config.HighlightPadding

	}

	return merged
}

func combineEvents_mergeted(segments []ClipSegment, config Config) []ClipSegment {
	if len(segments) == 0 {
		return nil
	}

	// Сортируем по времени начала
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Start < segments[j].Start
	})

	merged := []ClipSegment{segments[0]}

	for i := 1; i < len(segments); i++ {
		last := &merged[len(merged)-1]
		current := segments[i]

		// Объединяем если:
		// 1. Сегменты пересекаются (current.Start <= last.End)
		// 2. ИЛИ разрыв между сегментами меньше 1 секунды
		if current.Start <= last.End || current.Start-last.End < config.TransitionDuration {
			// Расширяем сегмент до конца текущего события
			if current.End > last.End {
				last.End = current.End
			}

			// Комбинируем тип события
			if last.Type != current.Type {
				last.Type = "combined"
			}

			// Увеличиваем оценку
			last.Score = (last.Score + current.Score) / 2

			// Объединяем детали
			if current.Details != "" {
				if last.Details != "" {
					last.Details += "; " + current.Details
				} else {
					last.Details = current.Details
				}
			}
		} else {
			// Добавляем новый сегмент
			merged = append(merged, current)
		}
	}

	return merged
}

func renderFinalVideo(inputPath string, segments []ClipSegment, outputPath string, config Config) error {
	if len(segments) == 0 {
		return fmt.Errorf("no segments to render")
	}

	absInputPath, _ := filepath.Abs(inputPath)
	absOutputPath, _ := filepath.Abs(outputPath)
	absTempDir, _ := filepath.Abs(config.TempDir)

	// Создание временной директории
	clipsDir := filepath.Join(absTempDir, "clips")
	if err := os.MkdirAll(clipsDir, 0750); err != nil {
		return fmt.Errorf("error creating clips directory: %w", err)
	}
	defer os.RemoveAll(clipsDir)

	var clipFiles []string
	var wg sync.WaitGroup
	var mu sync.Mutex
	errorOccurred := false

	// Канал для отслеживания прогресса
	progressChan := make(chan int, len(segments))
	go func() {
		completed := 0
		total := len(segments)
		for range progressChan {
			completed++
			percent := completed * 100 / total
			bars := percent / 2
			fmt.Printf("\rExtracting [%s%s] %d%%", 
				strings.Repeat("=", bars), 
				strings.Repeat(" ", 50-bars), 
				percent)
		}
		fmt.Println("\nSegments extraction complete")
	}()

	sem := make(chan struct{}, runtime.NumCPU()/4)

	// Извлечение сегментов с перекодированием
	for i, seg := range segments {
		if seg.End <= seg.Start {
			continue
		}

		wg.Add(1)
		go func(idx int, segment ClipSegment) {
			sem <- struct{}{}
			defer func() { <-sem }()
			defer wg.Done()
			clipPath := filepath.Join(clipsDir, fmt.Sprintf("clip_%04d.mp4", idx))
			duration := segment.End - segment.Start

			// Формируем базовые аргументы
			args := []string{
				"-loglevel", "warning",
				"-ss", fmt.Sprintf("%.6f", segment.Start),
				"-i", absInputPath,
				"-t", fmt.Sprintf("%.6f", duration),
				"-avoid_negative_ts", "make_zero",
				"-fflags", "+genpts",
				"-fps_mode", "cfr",
			}

			args = append(config.GPUAcceleration, args...)
			args = append(args, config.Codec_Args...)

			// Добавляем параметры кодирования
			args = append(args,
				"-x264-params", "keyint=30:min-keyint=30:scenecut=0",
				"-c:a", "aac",
				"-y",
				clipPath,
			)

			cmd := exec.Command("ffmpeg", args...) // #nosec G204
			if output, err := cmd.CombinedOutput(); err != nil {
				log.Printf("Clip extraction failed for segment %d: %v\n%s", idx, err, string(output))
				mu.Lock()
				errorOccurred = true
				mu.Unlock()
				return
			}

			if _, err := os.Stat(clipPath); err == nil {
				mu.Lock()
				clipFiles = append(clipFiles, clipPath)
				mu.Unlock()
			}
			progressChan <- 1
		}(i, seg)
	}

	wg.Wait()
	close(progressChan)
	if errorOccurred {
		log.Println("Warning: some segments failed extraction")
	}
	if len(clipFiles) == 0 {
		return fmt.Errorf("no valid segments extracted")
	}

	// Сортировка файлов по порядку
	sort.Slice(clipFiles, func(i, j int) bool {
		numI := regexp.MustCompile(`clip_(\d+)`).FindStringSubmatch(filepath.Base(clipFiles[i]))[1]
		numJ := regexp.MustCompile(`clip_(\d+)`).FindStringSubmatch(filepath.Base(clipFiles[j]))[1]
		return numI < numJ
	})

	// Создание списка файлов
	listFile := filepath.Join(absTempDir, "input.txt")
	cleanListFile := filepath.Clean(listFile)
	file, err := os.Create(cleanListFile)
	if err != nil {
		return fmt.Errorf("error creating list file: %w", err)
	}
	defer os.Remove(cleanListFile)
	
	for _, clip := range clipFiles {
		absClip, _ := filepath.Abs(clip)
		fmt.Fprintf(file, "file '%s'\n", absClip)
	}
	if err := file.Close(); err != nil {
		log.Printf("Error closing file %s: %v", cleanListFile, err)
	}

	// Сборка финального видео
	args := []string{
		"-f", "concat",
		"-safe", "0",
		"-i", cleanListFile,
		"-c", "copy",           // Безопасное копирование после перекодирования
		"-fflags", "+genpts",    // Восстановление временных меток
		"-y",
		absOutputPath,
	}

	fmt.Print("Assembling final video...")
	cmd := exec.Command("ffmpeg", args...) // #nosec G204
	if output, err := cmd.CombinedOutput(); err != nil {
		log.Printf("Assembly failed: %v\n%s", err, string(output))
		return fmt.Errorf("video assembly failed: %w", err)
	}
	fmt.Println("\rFinal video assembly completed")
	return nil
} 

// Генерация превью-видео (короткая версия)
func generatePreview(inputPath, outputPath string) error {
	// Получаем длительность видео
	duration, err := getVideoDuration(inputPath)
	if err != nil {
		return err
	}

	// Создаем 30-секундное превью
	previewDuration := 30.0
	if duration < previewDuration {
		previewDuration = duration
	}

	// Ускоряем видео в 4 раза
	speedFactor := 4.0

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, // #nosec G204
		"ffmpeg",
		"-i", inputPath,
		"-t", fmt.Sprintf("%.2f", previewDuration),
		"-vf", fmt.Sprintf("setpts=%.2f*PTS", 1/speedFactor),
		"-af", fmt.Sprintf("atempo=%.2f", speedFactor),
		"-preset", "veryfast",
		"-c:a", "aac",
		"-b:a", "64k",
		"-y",
		outputPath,
	)

	log.Println("Generating preview:", cmd.String())

	if output, err := cmd.CombinedOutput(); err != nil {
		log.Printf("Preview generation error:\n%s", string(output))
		return err
	}

	return nil
}

// Сохранение информации о сегментах
func saveSegments(segments []ClipSegment, path string) error {
	data, err := json.MarshalIndent(segments, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling segments: %w", err)
	}

	cleanPath := filepath.Clean(path)
	if err := os.WriteFile(cleanPath, data, 0600); err != nil {
		return fmt.Errorf("error writing segments file: %w", err)
	}

	return nil
}

func getSpeechCachePath(videoPath string, config Config) string {
	hash := sha256.Sum256([]byte(videoPath + fmt.Sprintf("%.2f|%.2f", config.SpeechThresholdMultiplier, config.MinSpeechDuration)))
	return filepath.Join(config.TempDir, fmt.Sprintf("speech_activity_%x.json", hash))
}

// Детекция речевой активности с помощью WebRTC VAD
func detectSpeechActivity(videoPath string, config Config) []ClipSegment {
	cachePath := getSpeechCachePath(videoPath, config)
	cleanCachePath := filepath.Clean(cachePath)

	// Попытка загрузить из кэша
	if data, err := os.ReadFile(cleanCachePath); err == nil {
		var cached []ClipSegment
		if err := json.Unmarshal(data, &cached); err == nil {
			log.Printf("Loaded cached speech segments: %d from %s", len(cached), cleanCachePath)
			return cached
		} else {
			log.Printf("Failed to parse speech cache: %v", err)
		}
	}

	// Создаем временный RAW файл
	tempRaw := filepath.Join(config.TempDir, "audio_temp.raw")
	cleanTempRaw := filepath.Clean(tempRaw)
	defer func() {
		if err := os.Remove(cleanTempRaw); err != nil {
			log.Printf("Error removing temp raw file: %v", err)
		}
	}()

	cmd := exec.Command( // #nosec G204
		"ffmpeg",
		"-i", videoPath,
		"-ac", "1",
		"-ar", "16000",
		"-af", "loudnorm,highpass=f=80,lowpass=f=3000",
		"-acodec", "pcm_s16le",
		"-f", "s16le",
		"-y",
		cleanTempRaw,
	)
	if err := cmd.Run(); err != nil {
		log.Printf("Audio conversion error: %v", err)
		return nil
	}

	data, err := os.ReadFile(cleanTempRaw)
	if err != nil || len(data) == 0 {
		log.Printf("Error reading RAW file or empty data")
		return nil
	}

	samples := make([]int16, len(data)/2)
	for i := 0; i < len(samples); i++ {
		samples[i] = int16(data[i*2]) | int16(data[i*2+1])<<8
	}

	sampleRate := 16000
	frameSize := sampleRate / 50
	minSpeechDuration := config.MinSpeechDuration
	minSilenceDuration := config.MinSilenceDuration

	var energies []float64
	for i := 0; i < len(samples); i += frameSize {
		end := i + frameSize
		if end > len(samples) {
			end = len(samples)
		}
		sumSquares := 0.0
		for j := i; j < end; j++ {
			sample := float64(samples[j]) / 32768.0
			sumSquares += sample * sample
		}
		rms := math.Sqrt(sumSquares / float64(end-i))
		energies = append(energies, rms)
	}

	if len(energies) == 0 {
		log.Println("No energy values calculated")
		return nil
	}

	sortedEnergies := append([]float64(nil), energies...)
	sort.Float64s(sortedEnergies)
	noiseIndex := int(float64(len(sortedEnergies)) * 0.1)
	if noiseIndex < 0 {
		noiseIndex = 0
	}
	noiseLevel := sortedEnergies[noiseIndex]
	speechThreshold := noiseLevel * config.SpeechThresholdMultiplier
	if speechThreshold < 0.005 {
		speechThreshold = 0.005
	}

	minSpeechFrames := int(math.Ceil(minSpeechDuration * float64(sampleRate) / float64(frameSize)))
	minSilenceFrames := int(math.Ceil(minSilenceDuration * float64(sampleRate) / float64(frameSize)))

	var segments []ClipSegment
	inSpeech := false
	speechStart := 0.0
	silentCounter := 0
	speechCounter := 0

	for frameIdx, energy := range energies {
		if energy > speechThreshold {
			silentCounter = 0
			speechCounter++
			if !inSpeech && speechCounter >= minSpeechFrames {
				inSpeech = true
				startFrame := frameIdx - minSpeechFrames + 1
				if startFrame < 0 {
					startFrame = 0
				}
				speechStart = float64(startFrame*frameSize) / float64(sampleRate)
			}
		} else {
			speechCounter = 0
			silentCounter++
			if inSpeech && silentCounter >= minSilenceFrames {
				inSpeech = false
				endFrame := frameIdx - silentCounter + 1
				if endFrame < 0 {
					endFrame = 0
				}
				endTime := float64(endFrame*frameSize) / float64(sampleRate)
				if endTime-speechStart >= minSpeechDuration {
					segments = append(segments, ClipSegment{
						Start: speechStart,
						End:   endTime,
						Type:  "speech",
						Score: 0.9,
					})
				}
				silentCounter = 0
			}
		}
	}

	if inSpeech {
		endTime := float64(len(samples)) / float64(sampleRate)
		if endTime-speechStart >= minSpeechDuration {
			segments = append(segments, ClipSegment{
				Start: speechStart,
				End:   endTime,
				Type:  "speech",
				Score: 0.9,
			})
		}
	}

	log.Printf("VAD stats: frames=%d, noise=%.5f, threshold=%.5f, segments=%d", len(energies), noiseLevel, speechThreshold, len(segments))

	// Сохраняем в кэш
	if jsonData, err := json.MarshalIndent(segments, "", "  "); err == nil {
		_ = os.WriteFile(cleanCachePath, jsonData, 0600)
	} else {
		log.Printf("Failed to write speech cache: %v", err)
	}

	return segments
}

// Комбинирование аудио-событий и речевой активности
func combineAudioEvents(audioEvents, speechEvents []ClipSegment) []ClipSegment {
	// Создаем временную шкалу событий
	events := append(audioEvents, speechEvents...)
	sort.Slice(events, func(i, j int) bool {
		return events[i].Start < events[j].Start
	})

	// Объединяем пересекающиеся события
	var merged []ClipSegment
	if len(events) == 0 {
		return merged
	}

	current := events[0]
	for i := 1; i < len(events); i++ {
		if events[i].Start <= current.End {
			if events[i].End > current.End {
				current.End = events[i].End
			}
			if events[i].Score > current.Score {
				current.Score = events[i].Score
			}
		} else {
			merged = append(merged, current)
			current = events[i]
		}
	}
	merged = append(merged, current)

	return merged
}