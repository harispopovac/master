<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Process;

class TranslationController extends Controller
{
    public function translate(Request $request)
    {
        $request->validate([
            'text' => 'required|string'
        ]);

        try {
            // Path to your Python script
            $pythonScript = base_path('../mmmmmm/src/test_translation.py');
            
            // Create a temporary file to store the input text
            $tempFile = tempnam(sys_get_temp_dir(), 'arabic_text_');
            file_put_contents($tempFile, $request->text);

            // Execute the Python script with the input text
            $result = Process::run("python3 {$pythonScript} --input {$tempFile}");

            // Clean up the temporary file
            unlink($tempFile);

            if ($result->successful()) {
                return response()->json([
                    'translation' => trim($result->output())
                ]);
            }

            return response()->json([
                'error' => 'Translation failed',
                'details' => $result->errorOutput()
            ], 500);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Translation failed',
                'details' => $e->getMessage()
            ], 500);
        }
    }
} 