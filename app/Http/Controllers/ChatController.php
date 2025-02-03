<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class ChatController extends Controller
{
    public function chat(Request $request)
    {
        $request->validate([
            'message' => 'required|string',
            'history' => 'array'
        ]);

        try {
            // Here you would typically make a call to your AI service
            // For now, we'll return a simple response
            $response = "I understand you said: " . $request->message . "\n\nI'm a simple echo bot for now, but I can be connected to an AI service like OpenAI's GPT to provide real responses.";

            return response()->json([
                'response' => $response
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Failed to process your request',
                'details' => $e->getMessage()
            ], 500);
        }
    }
} 