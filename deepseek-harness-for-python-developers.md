# Deepseek Harness - Cordis for Python Developers

下面用一套接近 Python 的伪代码，把 Cordis 的核心关系串起来。它不是可直接运行的实现，也没有复刻 TypeScript Proxy、异步竞态保护和诊断元数据，但保留 Cordis 的关键生命周期语义。

## 1. Context：插件运行时环境

```python
class Context:
    @classmethod
    def create_root(cls):
        """只在应用根部安装一次核心服务。"""
        raw = cls()
        raw.isolate_map = PrototypeMap()
        raw.intercept_map = PrototypeMap()

        # 真实 Cordis 返回由 ReflectService handler 驱动的 Proxy。
        proxy = ContextProxy(raw)
        raw.root = proxy
        raw.fiber = RootFiber(proxy)
        raw.reflect = ReflectService(proxy)
        raw.registry = PluginRegistry(proxy)
        raw.events = EventBus(proxy)
        raw.logger = Logger()
        return proxy

    def extend(self, metadata=None):
        """
        子 Context 原型继承当前 Context；不会重新创建
        Registry、EventBus 或 ReflectService。
        """
        child = prototype_inherit(self)
        define_own_properties(child, metadata or {})
        return child

    def get(self, name):
        """显式可选查找；服务由共享 ReflectService 按作用域解析。"""
        return self.reflect.get(self, name, strict=True)

    def provide(self, name, service):
        """注册由当前 Fiber 拥有、可自动撤销的服务。"""
        return self.reflect.provide(self, name, service)

    def plugin(self, plugin, config=None):
        """加载插件并返回可等待、可 dispose 的 Fiber。"""
        return self.registry.load(self, plugin, config or {})

    def on(self, event_name, listener):
        return self.events.on(self, event_name, listener)

    def effect(self, create_resource, label="anonymous"):
        # Effect 的真正所有者是当前 Fiber。
        return self.fiber.effect(create_resource, label)

    def isolate(self, service_name, label=None):
        labels = PrototypeMap(parent=self.isolate_map)
        labels[service_name] = label or unique_symbol(service_name)
        return self.extend({"isolate_map": labels})

    def intercept(self, service_name, config):
        configs = PrototypeMap(parent=self.intercept_map)
        configs[service_name] = config
        return self.extend({"intercept_map": configs})
```

真实 Cordis 中，`Context` 还是一个 Proxy。像：

```python
ctx.llm
ctx.tools
ctx.sessions
```

这样的属性访问会进入共享的 `ReflectService`，根据当前 Fiber 的依赖快照和服务 isolation label 解析，而不是从每个 Context 的独立字典读取。

`ctx.get("service")` 不要求当前插件声明 `inject`；默认 strict 查找只返回提供方 Fiber 已经 `ACTIVE` 的实现。相反，插件中的 `ctx.service` 属性访问必须经过依赖声明和当前 Fiber 的依赖快照，未声明依赖时会抛错。前者适合显式可选能力，后者用于必需依赖。

---

## 2. Plugin：由 Context 挂载的生命周期入口

DeepSeek Harness 不再定义一套独立于 Cordis 的插件基类，而是直接采用 Cordis 的结构化插件协议。可以把插件定义为：

> 能被 `ctx.plugin()` 挂载，由一个 Fiber 管理生命周期，并可向 Context 注册 Service、事件监听器、子插件或其他 Effect 的入口。

“结构化协议”意味着对象不必继承某个统一的 `Plugin` 类。当前 Python 实现接受三种入口形式。

### 函数插件

```python
def telemetry_plugin(ctx, config):
    ctx.on(
        "agent/status",
        lambda event: record_status(event),
    )


telemetry_plugin.inject = ["agents"]
fiber = await ctx.plugin(telemetry_plugin, config)
```

函数接收当前插件 Context 和经过校验的配置。它可以只注册监听器而不提供任何 Service。

### 类插件

```python
class LlmRuntime(Service):
    def __init__(self, ctx, config=None):
        super().__init__(ctx, "llm")
        self.adapters = {}


fiber = await ctx.plugin(LlmRuntime)
```

Registry 会把类构造成：

```python
LlmRuntime(plugin_context, validated_config)
```

`Service` 子类很适合充当类插件，但 Service 和插件不是同义词：Service 是挂在 `ctx.<name>` 上的能力；插件是创建并拥有这些运行时贡献的入口。

### 带 `apply()` 的对象或模块插件

当前 Python 版 DeepSeek Provider 就采用模块插件形式：

```python
# dsh.llm_deepseek 模块中的插件元数据
name = "llm-deepseek"
inject = ["llm"]


def Config(raw_config):
    return validate_and_apply_defaults(raw_config)


def apply(ctx, config):
    adapter = DeepSeekAdapter(config)
    ctx.llm.register_adapter(
        ["deepseek-official"],
        adapter,
    )
```

加载模块本身：

```python
import dsh.llm_deepseek as deepseek_plugin

fiber = await ctx.plugin(
    deepseek_plugin,
    {"apiKeyEnv": "DEEPSEEK_API_KEY"},
)
```

Registry 发现对象不可调用但存在可调用的 `apply`，就把 `apply` 作为该插件的 callback。

### 插件元数据

当前 Python Registry 实际读取三项可选元数据：

| 元数据 | 含义 |
|---|---|
| `name` | Fiber 诊断和显示名称 |
| `Config` | 在插件执行前校验并标准化配置的 callable |
| `inject` | 必需服务列表，或服务名到 intercept 配置的映射 |

上游 TypeScript Cordis 的插件类型还声明了 `provide` 和 `intercept` 元数据；当前 Python Registry 尚未消费这两项。真正的服务注册仍由 `Service.__init__()` 或 `ctx.provide()` 完成。

### 一次挂载如何成为运行中的插件

```text
ctx.plugin(plugin, raw_config)
  ↓
把函数、类或 apply 对象规范化为 callback
  ↓
按 callback 查找或创建共享 PluginRuntime
  ↓
为这次挂载创建独立 Fiber 和子 Context
  ↓
校验 Config，等待 inject 中的服务全部可用
  ↓
执行 callback(plugin_ctx, validated_config)
  ↓
收集 Service、监听器、子插件和 Effect
  ↓
依赖改变或 Fiber dispose 时撤销这些贡献
```

`PluginRuntime` 和 `Fiber` 需要区分：

```text
同一个插件 callback
  └── 一个共享 PluginRuntime
      ├── Fiber A：在 Context A 中的一次挂载
      ├── Fiber B：在 Context B 中的一次挂载
      └── Fiber C：使用另一份配置的一次挂载
```

`ctx.plugin()` 返回的是 Fiber，不是插件业务对象。Fiber 可等待到加载稳定，也可以通过 `dispose()` 卸载。

### 插件可以贡献什么

```python
def apply(ctx, config):
    # 提供直接能力
    ctx.provide("my_service", service)

    # 监听或拦截事件
    ctx.on("agent/status", listener)

    # 注册任意带清理函数的资源
    ctx.effect(start_resource)

    # 加载生命周期归当前插件所有的子插件
    ctx.plugin(ChildPlugin)

    # 向其他 Registry Service 注册贡献
    ctx.tools.register(tool)
    ctx.llm.register_adapter(provider_names, adapter)
```

这些注册都应成为当前 Fiber 的 Effect，使插件卸载后不会留下 Service、监听器、工具或 adapter。

最后，不要把几个相邻概念混为一谈：

| 概念 | 准确定义 |
|---|---|
| Plugin | 可由 `ctx.plugin()` 挂载的入口 |
| Service | 插件通过 `ctx.<name>` 暴露的能力 |
| Fiber | 一次插件挂载的状态、依赖快照和资源所有者 |
| Python 模块/发行包 | 代码分发单元，可以导出零个、一个或多个插件 |
| Bundle | 原 TypeScript dsh 中用于分发 Cordis 配置项和挂载代码的组合层 |

因此，“everything is a plugin”不是说每个类或文件都是插件，而是说 dsh 的产品行为都通过 Cordis 插件树参与组合和生命周期管理，没有绕过插件系统的特权产品内核。

---

## 3. Service：可被其他插件依赖的能力

一个 Service 可以理解为：

```text
有名字的长期运行对象
```

例如模型运行时服务：

```python
class LLMService:
    def stream(self, request):
        raise NotImplementedError
```

提供 `ctx.llm` 的是 LLM Runtime；DeepSeek 插件向 Runtime 注册 adapter，而不是替换整个服务：

```python
class LLMRuntime(LLMService):
    def __init__(self, ctx):
        self.ctx = ctx
        self.adapters = {}
        ctx.provide("llm", self)

    def register_adapter(self, name, adapter):
        def install():
            if name in self.adapters:
                raise Exception(f"adapter already exists: {name}")
            self.adapters[name] = adapter

            def dispose():
                del self.adapters[name]

            return dispose

        return self.ctx.effect(
            install,
            label=f'llm.register_adapter("{name}")',
        )


def DeepSeekProvider(ctx, config):
    ctx.llm.register_adapter(
        "deepseek",
        DeepSeekAdapter(config),
    )


DeepSeekProvider.inject = ["llm"]
```

这里把真实的 `registerAdapter(providerNames, adapter)` 简化成了单个名称；真实 API 可在一次注册中声明多个 provider 名。

加载后：

```python
await ctx.plugin(LLMRuntime)
await ctx.plugin(DeepSeekProvider, config)
```

其他插件可以使用：

```python
ctx.llm.stream(request)
```

在 dsh 中，通常还会用类型声明表达：

```typescript
interface Context {
    llm: LLMService
}
```

这样 TypeScript 能理解：

```typescript
ctx.llm.stream(request)
```

### Service 的关键特征

```text
Service
  ├── 有稳定名称，例如 "llm"
  ├── 注册到 Context
  ├── 可被其他插件查找
  ├── 有明确提供者
  └── 随提供它的插件一起卸载
```

因此 Service 不是普通的全局单例。

---

## 4. 插件注册表：管理插件实例

```python
class PluginRegistry:
    def __init__(self, root_context):
        self.root_context = root_context
        # 插件入口 callback -> 共享 Runtime
        self.runtimes = {}

    def load(self, parent_context, plugin, config):
        callback = resolve_callback(plugin)
        runtime = self.runtimes.setdefault(
            callback,
            PluginRuntime(
                callback=callback,
                config_schema=getattr(plugin, "Config", None),
            ),
        )

        fiber = Fiber(
            parent_context=parent_context,
            runtime=runtime,
            raw_config=config,
            inject=normalize_inject(
                getattr(plugin, "inject", None)
            ),
        )
        # 真实返回值既可 await，也可 dispose。
        return AwaitableFiber(fiber)

    def delete(self, plugin):
        callback = resolve_callback(plugin)
        runtime = self.runtimes.pop(callback, None)
        if runtime:
            for fiber in list(runtime.fibers):
                schedule(fiber.dispose())
```

注册表主要负责：

```text
加载插件
记录插件
等待插件依赖
启动插件
卸载插件
```

Cordis 支持函数插件、类插件，以及带 `apply()` 的对象插件。同一个插件入口可以被加载多次：

```python
ctx.plugin(MyPlugin, {"mode": "a"})
ctx.plugin(MyPlugin, {"mode": "b"})
```

这两次加载对应两个不同的 Fiber：

```text
MyPlugin
  ├── Fiber A / Config A
  └── Fiber B / Config B
```

插件代码可以相同，但作用域、配置和资源互相独立。

---

## 5. Fiber：一次插件运行实例

```python
class Fiber:
    def __init__(self, parent_context, runtime, raw_config, inject):
        self.parent = parent_context
        self.runtime = runtime
        self.raw_config = raw_config
        self.inject = inject

        # 子 Context 继承父 Context，只覆盖当前 Fiber。
        self.context = parent_context.extend({"fiber": self})

        self.effects = DisposableList()
        self.required_services = {}
        self.active_epoch = None
        self.uid = parent_context.registry.next_uid()
        self.state = "PENDING"

        # 子 Fiber 本身也是父 Fiber 拥有的 Effect。
        self.dispose = parent_context.fiber.effect(
            self._publish,
            label="ctx.plugin()",
        )

        self.check_dependencies()
        self.refresh()

    def _publish(self):
        remove_from_runtime = self.runtime.fibers.push(self)

        async def dispose_child():
            self.uid = None
            remove_from_runtime()
            await self.unload_plugin_body()
            self.state = "DISPOSED"

        return dispose_child

    def check_dependencies(self):
        for name in self.inject:
            implementation = self.context.reflect.get_implementation(
                self.context,
                name,
                strict=True,
            )
            if implementation is None:
                self.required_services.pop(name, None)
            else:
                self.required_services[name] = implementation

    def refresh(self):
        if any(
            name not in self.required_services
            for name in self.inject
        ):
            if self.state == "ACTIVE":
                schedule(self.unload_plugin_body())
            else:
                self.state = "PENDING"
            return

        epoch = tuple(
            self.required_services[name].owner_fiber.uid
            for name in self.inject
        )
        if self.state == "PENDING":
            schedule(self.load_plugin_body(epoch))
        elif epoch != self.active_epoch:
            schedule(self.reload_plugin_body(epoch))

    async def reload_plugin_body(self, epoch):
        await self.unload_plugin_body()
        await self.load_plugin_body(epoch)

    async def load_plugin_body(self, epoch):
        self.state = "LOADING"
        try:
            config = validate_config(
                self.runtime.config_schema,
                self.raw_config,
            )
            result = invoke_plugin(
                self.runtime.callback,
                self.context,
                config,
            )
            await self.collect_effect_result(result)
            self.active_epoch = epoch
            self.state = "ACTIVE"
            self.context.reflect.notify_owned_services(self)
        except Exception as error:
            self.error = error
            self.state = "FAILED"
            await self.rollback_partial_effects()
            raise

    async def unload_plugin_body(self):
        self.state = "UNLOADING"
        self.context.reflect.notify_owned_services(self)
        wrappers = self.effects.clear_reverse()
        await gather_settled(
            wrapper.dispose()
            for wrapper in wrappers
        )
        self.state = "PENDING"
```

Fiber 是资源所有权的核心：

```text
当前这个 Service 是谁提供的？
当前这个事件监听器是谁注册的？
当前这个 Timer 是谁创建的？
当前这个子插件属于谁？
```

答案都可以追溯到 Fiber。依赖缺失时 Fiber 不会立即抛错，而是保持 `PENDING`；服务出现后激活，服务消失或提供方改变后卸载并重新检查。配置校验或插件入口失败则进入 `FAILED`，不是回到 `PENDING`。

`ReflectService` 在服务注册、撤销或提供方活动状态变化时，会重新检查所有声明了对应 `inject` 的 Fiber，并调用其 `refresh()`。

---

## 6. 依赖注入：`inject`

例如本地 Bash Provider 依赖 subprocess：

```python
class LocalBash(ShellExecutor):
    inject = ["subprocess"]

    def __init__(self, ctx, config):
        # ShellExecutor.__init__ 最终注册 ctx.shell。
        super().__init__(ctx)
        self.subprocess = ctx.subprocess
        self.config = config

    def run(self, command):
        return self.subprocess.spawn(
            ["bash", "-c", command]
        )
```

加载顺序实际上由依赖关系决定：

```text
subprocess
    ↓
LocalBash
    ↓
shell tool
```

伪代码中的 `inject` 表示：

```python
class Plugin:
    inject = ["subprocess"]
```

Cordis 会等待 `subprocess` 可用后再启动该插件。若服务随后卸载或更换提供方，Fiber 会清理当前插件体，并在新依赖可用时重新执行。

这比手动写以下逻辑更可靠：

```python
while ctx.get("subprocess") is None:
    sleep(1)
```

---

## 7. Event System：插件之间通信

事件总线可以这样理解：

```python
class EventBus:
    def __init__(self, root_context):
        self.root_context = root_context
        self.listeners = {}

    def on(self, registering_context, event_name, listener,
           prepend=False, global_=False):
        owner = registering_context.fiber

        def install():
            hook = {
                "context": registering_context,
                "listener": listener,
                "global": global_,
            }
            hooks = self.listeners.setdefault(event_name, [])
            if prepend:
                hooks.insert(0, hook)
            else:
                hooks.append(hook)

            def dispose():
                if hook in hooks:
                    hooks.remove(hook)

            return dispose

        return owner.effect(
            install,
            label=f'ctx.on("{event_name}")',
        )

    def dispatch(self, event_name, this_arg=None):
        context_filter = get_context_filter(this_arg)
        return [
            hook["listener"]
            for hook in self.listeners.get(event_name, [])
            if (
                hook["global"]
                or context_filter is None
                or context_filter(hook["context"])
            )
        ]

    def emit(self, event_name, *args, this_arg=None):
        # 同步调用，不等待监听器返回的 awaitable。
        for listener in self.dispatch(event_name, this_arg):
            listener(*args)

    async def parallel(self, event_name, *args, this_arg=None):
        results = await gather_all_settled(
            listener(*args)
            for listener in self.dispatch(event_name, this_arg)
        )
        raise_aggregate_error_if_needed(results)

    async def serial(self, event_name, *args, this_arg=None):
        for listener in self.dispatch(event_name, this_arg):
            result = await listener(*args)
            if is_bail_value(result):
                return result

    def bail(self, event_name, *args, this_arg=None):
        for listener in self.dispatch(event_name, this_arg):
            result = listener(*args)
            if is_bail_value(result):
                return result
```

插件可以监听 Agent 状态：

```python
class TelemetryPlugin:
    def apply(self, ctx, config):
        ctx.on(
            "agent/status",
            lambda event: print(
                "agent status:",
                event["status"],
            ),
        )
```

Agent Loop 派发事件：

```python
ctx.emit(
    "agent/status",
    {
        "agent": agent,
        "status": "running",
    },
)
```

这样 Telemetry 插件不需要修改 Agent Loop。

---

## 8. Waterfall 事件：机制相同，载荷由事件定义

普通事件只是通知：

```python
ctx.emit("agent/status", payload)
```

Waterfall 更像中间件。以下是与具体事件无关的机制伪代码：

```python
def waterfall(event_name, args, final_handler):
    listeners = get_listeners(event_name)

    def next_():
        if listeners:
            listener = listeners.pop(0)
            return listener(*args, next_)
        return final_handler(*args)

    return next_()
```

调用 `next()` 委托给下一个监听器；不调用就截断后续链条。但每个 waterfall 能做什么，取决于该事件声明的载荷和返回类型，不能从通用机制推断。

### `agent/request` 只能替换模型调用配置

`agent/request` 的载荷只有 `agent`、`turn`、`step` 和 `signal`。它不携带 messages，也不返回模型响应流。监听器先取得 frozen 的 `LlmCallConfig`，再返回新的配置：

```python
async def route_model(payload, next):
    current = await next()
    return replace_frozen_config(
        current,
        provider="deepseek",
        model="deepseek-chat",
    )
```

下面的写法在 dsh 中是错误的：

```python
async def invalid_context_injection(payload, next):
    config = await next()
    config.messages.append("新的系统消息")  # 不存在且对象已冻结
    return config
```

`agent/request` 明确禁止修改模型可见消息，因为 dsh 要求：

```text
model-visible ⟺ logged
```

模型能看到的内容必须能从 Session 日志重建。

### 修改下一步消息使用 `agent/pre-step`

`agent/pre-step` 收到待进入步骤的 messages，并返回 `PreStepDecision`。插件可以在这里返回替换后的消息；Loop 接纳后会把进入请求的消息写入 Session：

```python
async def add_time_context(payload, next):
    decision = await next()
    if decision["kind"] == "reject":
        return decision

    time_message = make_identified_user_message(
        source="time-context",
        content=current_time_text(),
    )
    return {
        "kind": "enter",
        "messages": [
            *decision["messages"],
            time_message,
        ],
    }
```

也可以使用：

```python
agent.inject(context_message)
```

它把模型可见上下文排入 inbox，等待下一次 pre-step 领取，不会立即唤醒空闲 Agent。

权限也不是由 `agent/request` 实现的。工具权限由 `ctx.approval` 服务、`approval/request` waterfall，以及 `tools/pre-execute` 的决策桥接共同完成。完整模型请求的底层包裹点是 `llm/stream`，但其请求是只读的。

---

## 9. Effect：带清理逻辑的副作用

“副作用”指的是函数除了返回值之外，还改变了外部环境。

例如：

```python
timer = set_interval(...)
listener = event_bus.on(...)
process = spawn(...)
service = register_service(...)
```

这些都改变了 Context 之外或内部的状态。

普通写法：

```python
def start_plugin(ctx):
    timer = start_timer()

    # 如果忘记清理，就会泄漏
```

Effect 写法：

```python
def start_plugin(ctx):
    ctx.effect(create_timer)

def create_timer():
    timer = start_timer()

    def dispose():
        timer.stop()

    return dispose
```

生命周期变成：

```text
加载插件
  ↓
执行 create_timer()
  ↓
创建 timer
  ↓
保存 dispose()
  ↓
插件卸载
  ↓
执行 dispose()
  ↓
停止 timer
```

### 注册 Service 也是 Effect

```python
def provide(ctx, name, service):
    owner = ctx.fiber

    def install():
        implementation = register_in_reflect_store(
            context=ctx,
            name=name,
            value=service,
            owner=owner,
        )

        async def dispose():
            unregister_from_reflect_store(implementation)
            await notify_and_settle_dependents(name)

        return dispose

    return owner.effect(install)
```

### 注册监听器也是 Effect

```python
def on(ctx, event_name, listener):
    owner = ctx.fiber

    def install():
        hook = register_hook(
            context=ctx,
            event_name=event_name,
            listener=listener,
        )

        def dispose():
            unregister_hook(hook)

        return dispose

    return owner.effect(install)
```

所以：

```text
ctx.provide()
  = 加入服务 + 登记删除动作

ctx.on()
  = 加入监听器 + 登记移除动作

ctx.effect()
  = 直接登记任意资源的清理动作
```

一个 Effect 可以返回一个 disposer、异步产生 disposer，或者通过 iterable 产生多个 disposer。同一个 Effect 内产生的多个 disposer 按逆序串行执行。

Fiber 卸载时，顶层 Effect wrapper 会按逆序取出，但随后并发等待。因此下面三个独立 Effect 之间不能依赖严格的串行顺序：

```python
ctx.effect(create_database)
ctx.effect(create_cache)
ctx.effect(create_worker)
```

如果清理必须严格按顺序进行，应放入同一个复合 Effect：

```python
def create_related_resources():
    database = create_database()
    cache = create_cache(database)
    worker = create_worker(cache)

    # 同一个 Effect 内 disposer 逆序执行。
    yield lambda: database.close()
    yield lambda: cache.close()
    yield lambda: worker.stop()


ctx.effect(create_related_resources)
```

这样清理顺序才是 `worker → cache → database`。

---

## 10. 子插件和子 Context

父插件加载子插件：

```python
class WebApp:
    def apply(self, ctx, config):
        ctx.plugin(ApiPlugin)
        ctx.plugin(UiPlugin)
```

关系是：

```text
WebApp Fiber
  └── WebApp Context
        ├── ApiPlugin Fiber
        │     └── ApiPlugin Context
        └── UiPlugin Fiber
              └── UiPlugin Context
```

子 Context 原型继承父 Context，并共享根级 Reflect、Registry 与 Events。服务不是复制到子 Context 的：

```python
parent.provide("llm", deepseek_llm)

child = parent.extend({"fiber": child_fiber})

child.get("llm")
# 得到 deepseek_llm
```

但子插件注册的资源归自己所有：

```python
child.provide("temporary_service", service)
child.on("some-event", listener)
```

当子插件卸载：

```text
temporary_service 被删除
listener 被移除
子插件自己的 timers 被停止
子插件自己的进程被关闭
```

父 Context 不会受影响。子插件 Fiber 本身也是父 Fiber 拥有的 Effect，因此父插件卸载会触发子 Fiber 的完整卸载。

---

## 11. 服务隔离

有时子 Context 不想继承父 Context 的某个服务，而是需要自己的实现：

```python
child = parent.isolate("llm")

child.provide("llm", isolated_llm_runtime)
```

结果：

```text
父 Context:
  ctx.llm = LLMRuntime

子 Context:
  ctx.llm = IsolatedLLMRuntime
```

父 Context 中的其他服务仍然可以继承：

```text
子 Context:
  ctx.llm       -> IsolatedLLMRuntime
  ctx.sessions  -> 继承父级
  ctx.tools     -> 继承父级
  ctx.logger    -> 继承父级
```

这适合：

- 为不同 Agent 使用不同模型；
- 为测试替换 Provider；
- 为某个会话提供不同工具；
- 为子任务提供受限能力。

---

## 12. 用一段伪代码串起整个 dsh

```python
root = Context.create_root()

# 1. 注册底层能力
subprocess_fiber = await root.plugin(SubprocessProvider)
session_fiber = await root.plugin(SessionStore)

# 2. LLM Runtime 提供 ctx.llm，DeepSeek 插件注册 adapter
llm_fiber = await root.plugin(LLMRuntime)
deepseek_fiber = await root.plugin(DeepSeekProvider, {
    "model": "deepseek-chat",
})

# 3. 注册 Shell Provider
shell_fiber = await root.plugin(LocalBash, {
    "cwd": "/workspace",
})

# 4. 注册面向模型的工具
bash_tool_fiber = await root.plugin(BashTool)

# 5. 注册策略插件
approval_fiber = await root.plugin(ApprovalPolicy)
telemetry_fiber = await root.plugin(TelemetryPlugin)
time_context_fiber = await root.plugin(TimeContextPlugin)

# 6. 加载 Agent Loop Provider；返回的是 Fiber
agent_loop_fiber = await root.plugin(AgentLoopPlugin)

# 7. 通过 ctx.agents 的工厂入口取得带所有权的 AgentHandle
handle = await root.agents.create({
    "sessionId": "session-1",
})
agent = handle.agent

# UserMessage 必须带身份和来源；followup 会唤醒一个新 turn。
agent.followup(
    make_identified_user_message(
        source="human",
        content="解释当前项目",
    )
)
await agent.whenIdle()

# 8. dispose 属于 AgentHandle，不属于 Agent 接口
await handle.dispose()
await agent_loop_fiber.dispose()
```

`ctx.agentLoop.create(id)` 也可以同步返回一个裸 `Agent`，但其生命周期绑定到调用方 Fiber，没有公开的 `agent.dispose()`。需要调用方显式控制销毁时，应使用 `ctx.agents.create(...)` 返回的 `AgentHandle`。

最终依赖关系是：

```text
Context
  ├── Registry
  ├── SessionStore
  ├── LLMRuntime
  ├── DeepSeek adapter plugin
  ├── SubprocessProvider
  ├── LocalBash
  ├── BashTool
  ├── ApprovalPolicy
  ├── TelemetryPlugin
  └── AgentLoop
```

所有插件都通过 Context 连接，但各自只依赖自己声明的服务和事件。

## 最后总结

```text
Context
  = 当前插件作用域中的运行时环境

Plugin
  = 由 ctx.plugin() 挂载、由 Fiber 管理生命周期的结构化入口

Service
  = 通过名字暴露、可被依赖的能力

Plugin Registry
  = 管理插件加载和运行实例

Fiber
  = 一次插件运行及其资源所有权

Event System
  = 插件之间的类型化通信机制

Waterfall
  = 可委托或截断的事件链；允许替换什么由具体事件类型决定

Effect
  = 立即执行 setup、收集 disposer，并把清理绑定到当前 Fiber

Child Context
  = 通过 extend() 继承父 Context，并可 isolate 或 intercept

inject
  = 动态生命周期依赖；服务变化会驱动 Fiber 装卸或重载
```

Cordis 的核心价值就是把“对象如何被找到”和“资源何时被清理”统一到了 Context 和 Fiber 的生命周期中。
