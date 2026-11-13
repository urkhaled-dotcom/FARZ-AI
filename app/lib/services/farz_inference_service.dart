import 'dart:io';
import 'package:flutter/services.dart';
import '../models/classification_result.dart';

class FarzInferenceService {
  static const MethodChannel _channel = MethodChannel('farz/inference');

  Future<bool> isModelReady() async {
    try {
      return await _channel.invokeMethod<bool>('isModelReady') ?? false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  Future<ClassificationResult> classify(File image) async {
    if (!await image.exists()) throw Exception('Image file no longer exists: ${image.path}');
    final response = await _channel.invokeMethod<Map<dynamic, dynamic>>(
      'classifyImage', {'path': image.path},
    );
    if (response == null) {
      throw const PlatformException(code: 'EMPTY_RESULT', message: 'The on-device model returned no result.');
    }
    return ClassificationResult.fromPlatform(response);
  }
}
