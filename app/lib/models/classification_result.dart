class ClassScore {
  const ClassScore({required this.label, required this.score});
  final String label;
  final double score;
}

class ClassificationResult {
  const ClassificationResult({
    required this.label,
    required this.score,
    required this.latencyMs,
    required this.scores,
  });
  final String label;
  final double score;
  final double latencyMs;
  final List<ClassScore> scores;

  factory ClassificationResult.fromPlatform(Map<dynamic, dynamic> data) {
    const names = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal'];
    final rawScores = data['scores'] as List<dynamic>? ?? const [];
    final values = <ClassScore>[];
    for (var i = 0; i < rawScores.length && i < names.length; i++) {
      values.add(ClassScore(label: names[i], score: (rawScores[i] as num).toDouble()));
    }
    values.sort((a, b) => b.score.compareTo(a.score));
    final index = (data['predicted_index'] as num?)?.toInt() ?? 0;
    final label = index >= 0 && index < names.length ? names[index] : 'Unknown';
    final score = values.where((item) => item.label == label).firstOrNull?.score ?? 0.0;
    return ClassificationResult(
      label: label,
      score: score,
      latencyMs: (data['latency_ms'] as num?)?.toDouble() ?? 0,
      scores: values,
    );
  }
}
