// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Curated from NVIDIA's public LLM API catalog on 2026-03-09:
// https://docs.api.nvidia.com/nim/reference/llm-apis
const AVAILABLE_CHAT_MODELS = [
  'deepseek-ai/deepseek-v3.1',
  'deepseek-ai/deepseek-v3.1-terminus',
  'deepseek-ai/deepseek-v3.2',
  'minimaxai/minimax-m2.1',
  'minimaxai/minimax-m2.5',
  'mistralai/devstral-2-123b-instruct-2512',
  'moonshotai/kimi-k2-instruct',
  'moonshotai/kimi-k2-instruct-0905',
  'moonshotai/kimi-k2-thinking',
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'nvidia/llama-3.3-nemotron-super-49b-v1',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'nvidia/nvidia-nemotron-nano-9b-v2',
  'openai/gpt-oss-20b',
  'openai/gpt-oss-120b',
  'qwen/qwen3-coder-480b-a35b-instruct',
  'qwen/qwen3-next-80b-a3b-instruct',
  'qwen/qwen3-next-80b-a3b-thinking',
  'z-ai/glm4.7',
  'z-ai/glm5'
];

// Alias mapping for OpenAI- or Janitor-style model names.
const MODEL_ALIASES = {
  'gpt-3.5-turbo': 'nvidia/nvidia-nemotron-nano-9b-v2',
  'gpt-4': 'deepseek-ai/deepseek-v3.1',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.2',
  'gpt-4o-mini': 'openai/gpt-oss-20b',
  'claude-3-opus': 'moonshotai/kimi-k2-thinking',
  'claude-3-sonnet': 'openai/gpt-oss-120b',
  'claude-3.5-sonnet': 'openai/gpt-oss-120b',
  'claude-3.5-haiku': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-instruct',
  'gemini-1.5-pro': 'qwen/qwen3-next-80b-a3b-instruct',
  'gemini-1.5-flash': 'nvidia/nvidia-nemotron-nano-9b-v2',
  'gemini-2.0-flash': 'nvidia/nvidia-nemotron-nano-9b-v2'
};

function sendOpenAIError(res, status, message, type = 'invalid_request_error') {
  return res.status(status).json({
    error: {
      message,
      type,
      code: status
    }
  });
}

function pickDefinedFields(source, fields) {
  return fields.reduce((result, field) => {
    const value = source[field];

    if (value !== undefined && value !== null) {
      result[field] = value;
    }

    return result;
  }, {});
}

function validateChatCompletionRequest(body = {}) {
  const {
    model,
    messages,
    temperature,
    max_tokens,
    stream,
    top_p,
    presence_penalty,
    frequency_penalty,
    seed,
    stop
  } = body;

  if (typeof model !== 'string' || model.trim() === '') {
    return 'The `model` field is required and must be a non-empty string.';
  }

  if (!Array.isArray(messages)) {
    return 'The `messages` field is required and must be an array.';
  }

  if (temperature !== undefined && (!Number.isFinite(temperature))) {
    return 'The `temperature` field must be a finite number when provided.';
  }

  if (max_tokens !== undefined && (!Number.isInteger(max_tokens) || max_tokens < 0)) {
    return 'The `max_tokens` field must be a non-negative integer when provided.';
  }

  if (stream !== undefined && typeof stream !== 'boolean') {
    return 'The `stream` field must be a boolean when provided.';
  }

  if (top_p !== undefined && top_p !== null && !Number.isFinite(top_p)) {
    return 'The `top_p` field must be a finite number when provided.';
  }

  if (presence_penalty !== undefined && presence_penalty !== null && !Number.isFinite(presence_penalty)) {
    return 'The `presence_penalty` field must be a finite number when provided.';
  }

  if (frequency_penalty !== undefined && frequency_penalty !== null && !Number.isFinite(frequency_penalty)) {
    return 'The `frequency_penalty` field must be a finite number when provided.';
  }

  if (seed !== undefined && seed !== null && !Number.isInteger(seed)) {
    return 'The `seed` field must be an integer when provided.';
  }

  if (
    stop !== undefined &&
    stop !== null &&
    !(typeof stop === 'string' || (Array.isArray(stop) && stop.every(item => typeof item === 'string')))
  ) {
    return 'The `stop` field must be a string or an array of strings when provided.';
  }

  return null;
}

function resolveNimModel(model) {
  if (MODEL_ALIASES[model]) {
    return MODEL_ALIASES[model];
  }

  // Direct NIM model ids are namespaced (for example `meta/...`), so pass them through.
  if (model.includes('/')) {
    return model;
  }

  const modelLower = model.toLowerCase();

  if (modelLower.includes('coder') || modelLower.includes('code')) {
    return 'qwen/qwen3-coder-480b-a35b-instruct';
  }

  if (
    modelLower.includes('think') ||
    modelLower.includes('reason') ||
    modelLower.includes('opus') ||
    modelLower.includes('o1') ||
    modelLower.includes('o3')
  ) {
    return 'moonshotai/kimi-k2-thinking';
  }

  if (modelLower.includes('mini') || modelLower.includes('haiku') || modelLower.includes('flash')) {
    return 'nvidia/nvidia-nemotron-nano-9b-v2';
  }

  if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('sonnet')) {
    return 'openai/gpt-oss-120b';
  }

  return 'deepseek-ai/deepseek-v3.2';
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = [...new Set([...Object.keys(MODEL_ALIASES), ...AVAILABLE_CHAT_MODELS])].map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: model.includes('/') ? model.split('/')[0] : 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const validationError = validateChatCompletionRequest(req.body);
    if (validationError) {
      return sendOpenAIError(res, 400, validationError);
    }

    const {
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;
    const model = req.body.model.trim();
    const nimModel = resolveNimModel(model);
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      ...pickDefinedFields(req.body, [
        'top_p',
        'stop',
        'frequency_penalty',
        'presence_penalty',
        'seed'
      ]),
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream ?? false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);

    return sendOpenAIError(
      res,
      error.response?.status || 500,
      error.message || 'Internal server error'
    );
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  return sendOpenAIError(res, 404, `Endpoint ${req.path} not found`);
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
