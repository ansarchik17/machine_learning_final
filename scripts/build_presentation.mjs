/** Editable defense slides from the project's frozen numerical evidence. */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RUNTIME = process.env.PRESENTATION_RUNTIME_DIR ?? path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime');
const MODULES = path.join(RUNTIME, 'dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES = MODULES;
const requireRuntime = createRequire(path.join(MODULES, '__presentation__.cjs'));
const {Presentation, PresentationFile} = await import(pathToFileURL(requireRuntime.resolve('@oai/artifact-tool')).href);
const SKILL = process.env.PRESENTATION_SKILL_DIR ?? path.join(os.homedir(), '.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11011/skills/presentations');
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const BUILD = path.join(ROOT, 'tmp/presentation');
const OUT = path.join(ROOT, 'output/presentation');
await fs.mkdir(BUILD, {recursive:true});
await fs.mkdir(OUT, {recursive:true});
const data = JSON.parse(await fs.readFile(path.join(ROOT,'artifacts/presentation_content.json'),'utf8'));
const deck = Presentation.create({slideSize:{width:1280,height:720}});
const C = {bg:'#F7F8FA',ink:'#111827',muted:'#566575',teal:'#087F8C',orange:'#B45309',blue:'#2563EB',grid:'#D9E0E7'};
const FONT = 'Arial';
const fontPolicy = {basis:'design',families:[FONT]};
const classColors = [C.orange,C.blue,C.teal];
const fmt = (value,n=3) => Number(value).toFixed(n).replace('.',',');

function text(s,value,x,y,w,h,size=28,color=C.ink,bold=false,align='left') {
  const sh = s.shapes.add({geometry:'textbox',name:`text-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=String(value);
  sh.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',insets:0,autoFit:'none',wrap:'square'};
  return sh;
}
function slide(number,{dark=false}={}) {
  const s=deck.slides.add();s.background.fill=dark?C.ink:C.bg;
  const meta=data.slides[number-1];
  if(number!==1) text(s,meta.title,64,42,1152,110,42,dark?'#FFFFFF':C.ink,true);
  text(s,`${number} / 15`,1110,675,105,25,18,dark?'#B8C9D5':C.muted,false,'right');
  s.speakerNotes.text=`Предлагаемый докладчик: ${meta.speaker}. ${meta.seconds} секунд. Распределение выступления не подтверждает технический вклад.\n\n${meta.spoken_notes}\n\nИсточники (не зачитывать):\n${meta.sources.join('\n')}`;
  return s;
}
function foot(s,value,y=628) {text(s,value,64,y,1152,42,20,C.muted);}
function chart(s,kind,options) {
  // Excel stores at most 15 significant digits. Retain twelve in the editable
  // chart workbook; full-precision experimental values remain in the sources.
  const portableOptions={...options,series:options.series.map(series=>({
    ...series,values:series.values.map(value=>value==null?null:Number(value.toPrecision(12))),
  }))};
  const result=s.charts.add(kind,{
    chartFill:'none',chartLine:{fill:'none',width:0},plotAreaFill:'none',plotAreaLine:{fill:'none',width:0},
    hasLegend:false,
    xAxis:{textStyle:{typeface:FONT,fontSize:23,fill:C.ink},line:{fill:C.grid,width:1},majorGridlines:null},
    yAxis:{textStyle:{typeface:FONT,fontSize:23,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.grid,width:1}},
    ...portableOptions,
  });
  applyPresentationChartFont(result,{fontFamily:FONT});
  return result;
}
function table(s,values,x,y,w,h,widths,{fontSize=25,headerSize=22}={}) {
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,
    columnTracks:widths.map(value=>({mode:'fr',value}))});
  t.borders.assign({fill:C.bg,width:0,style:'solid'});
  t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
    textStyle:{typeface:FONT,fontSize,color:C.ink},margins:{left:12,right:10,top:13,bottom:10},
  });
  t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.teal,textStyle:{typeface:FONT,fontSize:headerSize,bold:true,color:'#FFFFFF'}});
  for(let r=1;r<values.length;r++) {
    t.rows[r].height=h/values.length;
    for(let c=0;c<values[0].length;c++) t.getCell(r,c).fill=r%2?'#FFFFFF':'#EDF2F5';
  }
  t.rows[0].height=h/values.length;
  return t;
}
function node(s,title,subtitle,x,y,w,h) {
  const sh=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:C.teal,width:2}});
  text(s,title,x+18,y+22,w-36,70,52,C.teal,true,'center');
  text(s,subtitle,x+16,y+105,w-32,h-110,26,C.ink,false,'center');
  return sh;
}
function link(s,from,to) {
  s.shapes.connect(from,to,{kind:'straight',fromSide:'right',toSide:'left',line:{fill:C.muted,width:2},tail:{type:'triangle',width:'med',length:'med'}});
}

// 1. Minimal typographic cover.
{
 const s=slide(1,{dark:true});
 text(s,'Финальный проект по машинному обучению',72,56,1120,34,23,'#B8C9D5');
 text(s,'Классификация образовательных\nисходов студентов и анализ\nакадемических профилей',72,164,1140,220,54,'#FFFFFF',true);
 text(s,'Student Outcome Classification and Academic Profile Analysis',72,422,1130,38,25,'#B8C9D5');
 text(s,'Ансар Келес, Мухамеджан Сагым, Бекназар Сарсенбек',72,557,1120,40,25,'#FFFFFF');
 text(s,'UCI 697, три исхода обучения',72,615,1050,30,22,'#84CAD0');
}
// 2. Research question and literature establish the two comparisons.
{
 const s=slide(2);
 text(s,'Как семестровые результаты меняют\nкачество классификации?',64,157,1140,100,38,C.teal,true);
 text(s,'Алгоритмы',64,310,440,44,31,C.ink,true);
 text(s,'Одинаковые строки и признаки\nDummy, LR, Tree, KNN, ANN',64,372,510,104,29);
 text(s,'Состав информации',680,310,500,44,31,C.ink,true);
 text(s,'Одинаковая Logistic Regression\nБез семестров, первый, оба',680,372,536,104,29);
 text(s,'Литература связывает качество с этапом обучения\nи требует отделять выбор модели от итоговой оценки',64,520,1152,78,26,C.muted);
 foot(s,'Glandorf et al. (2024), Cawley & Talbot (2010)');
}
// 3. Overall class distribution is distinct from train-only EDA.
{
 const s=slide(3);const counts=[1421,794,2209];
 text(s,'4 424 студента',64,157,750,64,44,C.teal,true);
 chart(s,'bar',{position:{left:64,top:244,width:755,height:365},categories:['Dropout','Enrolled','Graduate'],
   series:[{name:'Студенты',values:counts,points:counts.map((_,idx)=>({idx,fill:classColors[idx]}))}],
   barOptions:{direction:'column',gapWidth:100},yAxis:{min:0,max:2500,majorUnit:500,numberFormatCode:'0',textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0',textStyle:{fontSize:26,bold:true,fill:C.ink}}});
 text(s,'36 исходных признаков\nОдин португальский вуз',880,165,335,90,28);
 text(s,'Dropout, 32,12%\nПрекращение программы',880,294,335,84,25,C.orange);
 text(s,'Enrolled, 17,95%\nОбучение продолжается',880,401,335,84,25,C.blue);
 text(s,'Graduate, 49,93%\nПрограмма завершена',880,508,335,84,25,C.teal);
 foot(s,'Весь набор. UCI 697, Realinho et al. (2021, 2022), CC BY 4.0');
}
// 4. Explicit editable protocol diagram with no feedback from test.
{
 const s=slide(4);text(s,'Общий stratified split 70 / 15 / 15, seed 42',64,158,1150,45,29,C.muted);
 const train=node(s,'3 096','Train\nОбучение и 5-fold CV',64,262,340,215);
 const validation=node(s,'664','Validation\nВыбор кандидата и эпохи',470,262,340,215);
 const test=node(s,'664','Test\nИтог после фиксации',876,262,340,215);
 link(s,train,validation);link(s,validation,test);
 text(s,'Preprocessing обучается внутри train и каждого CV-fold',64,532,1152,43,29,C.teal,true);
 foot(s,'Test не участвует в подборе. Повтор того же split проверяет воспроизводимость');
}
// 5. Feature counts and preparation remain editable text.
{
 const s=slide(5);
 text(s,'28 + 4 = 32',64,178,650,88,68,C.teal,true);
 text(s,'Исходные и производные признаки\nосновного профиля первого семестра',64,281,670,95,30);
 text(s,'229',864,178,350,88,68,C.teal,true);
 text(s,'Входов после\none-hot encoding',864,281,350,95,30);
 text(s,'Категориальные коды и Unknown\nСтандартизация для LR, KNN и ANN\nОбучаемые преобразования только на train',64,432,1152,132,29);
 foot(s,'Второй семестр и два финансовых статуса исключены. Нули и все строки сохранены');
}
// 6. Actual means on the training subset, with no invented uncertainty.
{
 const s=slide(6);const values=data.training_profiles.map(row=>row.mean_passed_sem1);
 chart(s,'bar',{position:{left:64,top:170,width:770,height:425},categories:data.training_profiles.map(row=>row.Target),
   series:[{name:'Среднее сданных дисциплин',values,points:values.map((_,idx)=>({idx,fill:classColors[idx]}))}],
   barOptions:{direction:'column',gapWidth:120},yAxis:{min:0,max:7,majorUnit:1,title:{text:'Среднее число дисциплин',textStyle:{fontSize:22}},textStyle:{fontSize:23,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},
   dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:28,bold:true,fill:C.ink}}});
 text(s,'Первый семестр',890,191,326,46,29,C.teal,true);
 text(s,'Средние различаются\nмежду исходами',890,269,326,96,31);
 text(s,'Связь описательная.\nРаспределения\nперекрываются',890,426,326,140,27,C.muted);
 foot(s,'Только train, n = 3 096. Источник: training_class_profiles.csv');
}
// 7. Native table of frozen classical CV choices.
{
 const s=slide(7);
 table(s,[['Модель','Лучшие параметры','CV Macro F1'],
  ['Logistic Regression','C = 0.1, balanced','0,648 ± 0,005'],
  ['Decision Tree','depth = 3, leaf = 5, balanced','0,611 ± 0,022'],
  ['KNN','k = 15, uniform','0,561 ± 0,018']],64,176,1152,290,[1.35,2.2,1.1],{fontSize:25});
 text(s,'Macro F1 даёт одинаковый вес трём классам',64,510,1152,45,31,C.teal,true);
 text(s,'Dummy задаёт частотный baseline. CV использует только train',64,567,1152,44,27);
 foot(s,'Среднее ± SD по 5 folds. Проверено 6 / 16 / 6 конфигураций');
}
// 8. Native, connected and editable ANN architecture.
{
 const s=slide(8);
 const a=node(s,'229','Подготовленные\nвходы',64,200,300,230);
 const b=node(s,'32','Скрытый слой\nReLU',490,200,300,230);
 const c=node(s,'3','Класса\nsoftmax',916,200,300,230);
 link(s,a,b);link(s,b,c);
 text(s,'7 459 параметров',64,481,1152,56,40,C.teal,true);
 text(s,'Cross-entropy, Adam, L2 = 0.01, веса классов',64,549,1152,45,29);
 foot(s,'Learning rate 0.001, batch 64. Четыре кандидата, выбор по validation');
}
// 9. All saved epochs of the selected ANN, not an interpolated curve.
{
 const s=slide(9);const epochs=data.history.map(row=>String(row.epoch));
 chart(s,'line',{position:{left:64,top:175,width:815,height:420},categories:epochs,
  series:[{name:'Train',values:data.history.map(r=>r.train_macro_f1),line:{fill:C.teal,width:3},marker:{symbol:'none'}},
          {name:'Validation',values:data.history.map(r=>r.validation_macro_f1),line:{fill:C.orange,width:3},marker:{symbol:'none'}}],
  hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:23}},
  xAxis:{title:{text:'Эпоха',textStyle:{fontSize:22}},textStyle:{fontSize:16,fill:C.muted},majorGridlines:null},
  yAxis:{min:0.5,max:0.85,majorUnit:0.05,title:{text:'Macro F1',textStyle:{fontSize:23}},numberFormatCode:'0.00',textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:C.grid,width:1}}});
 const best=data.history.find(row=>row.epoch===9),last=data.history.at(-1);
 text(s,'Checkpoint 9',930,195,286,45,30,C.teal,true);
 text(s,`Train ${fmt(best.train_macro_f1)}\nValidation ${fmt(best.validation_macro_f1)}`,930,256,286,90,27);
 text(s,'Остановка 34',930,392,286,45,30,C.orange,true);
 text(s,`Train ${fmt(last.train_macro_f1)}\nValidation ${fmt(last.validation_macro_f1)}`,930,453,286,90,27);
 foot(s,'34 реальные эпохи. Увеличенная шкала F1: 0,50–0,85. Восстановлены веса эпохи 9');
}
// 10. Same-feature algorithm comparison, using both original metric series.
{
 const s=slide(10);
 chart(s,'bar',{position:{left:64,top:169,width:1152,height:415},categories:['Dummy','LR','Tree','KNN','ANN'],
  series:[{name:'Validation',values:data.models.map(r=>r.validation_macro_f1),fill:C.teal},
          {name:'Test',values:data.models.map(r=>r.test_macro_f1),fill:C.blue}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:70},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:23}},
  yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.0',title:{text:'Macro F1',textStyle:{fontSize:23}},textStyle:{fontSize:23,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:21,fill:C.ink}}});
 text(s,'ANN: validation 0,6625',64,595,540,40,29,C.teal,true);
 text(s,'LR: test 0,6899',680,595,536,40,29,C.blue,true);
 foot(s,'Первый семестр, 229 входов. Общие 664 test-строки. Выбор модели сохранён',645);
}
// 11. Fixed-algorithm feature comparison, never mixed with the preceding ranking.
{
 const s=slide(11);const values=data.ablation.map(row=>row.test_macro_f1);
 chart(s,'bar',{position:{left:64,top:175,width:850,height:430},categories:['Без семестров','Первый семестр','Оба семестра'],
  series:[{name:'Test Macro F1',values,fill:C.teal}],barOptions:{direction:'column',gapWidth:125},
  yAxis:{min:0,max:1,majorUnit:0.2,numberFormatCode:'0.0',title:{text:'Test Macro F1',textStyle:{fontSize:23}},textStyle:{fontSize:23,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:27,bold:true,fill:C.ink}}});
 text(s,'Logistic\nRegression',962,191,254,98,31,C.teal,true);
 text(s,`+${fmt(values[1]-values[0])}\nПервый семестр`,962,337,254,95,28);
 text(s,`+${fmt(values[2]-values[1])}\nВторой семестр`,962,478,254,95,28);
 foot(s,'Общие строки, grid и CV. Различается информация, её доступность по датам не установлена');
}
// 12. Cluster table and PCA variance use five train-only academic variables.
{
 const s=slide(12);
 text(s,'K-Means, k = 3',64,168,530,46,31,C.teal,true);
 const profiles=data.cluster_profiles;
 table(s,[['Профиль','n','Сдано¹'],
  ['Обычная нагрузка',String(profiles[0].n),fmt(profiles[0]['mean_Curricular units 1st sem (approved)'],2)],
  ['Почти нет сданных',String(profiles[1].n),'≈ 0'],
  ['Высокая нагрузка',String(profiles[2].n),fmt(profiles[2]['mean_Curricular units 1st sem (approved)'],2)]],64,234,540,270,[2.2,.85,.85],{fontSize:24,headerSize:23});
 text(s,'Silhouette 0,600\nМинимальная группа 5,91%',64,537,540,78,27);
 text(s,'PCA, накопленная дисперсия',665,168,551,46,29,C.teal,true);
 chart(s,'line',{position:{left:665,top:232,width:551,height:318},categories:['1','2','3','4','5'],
  series:[{name:'Накопленная дисперсия',values:data.pca.map(r=>r.cumulative_variance),line:{fill:C.teal,width:3},marker:{symbol:'circle',size:7,fill:C.teal}}],
  yAxis:{min:0,max:1,majorUnit:0.25,numberFormatCode:'0%',textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:C.grid,width:1}},
  xAxis:{title:{text:'Компоненты',textStyle:{fontSize:22}},textStyle:{fontSize:23,fill:C.ink}}});
 text(s,'2 компоненты: 77,80%\n4 компоненты: 97,92%',715,554,501,70,27,C.teal,true);
 foot(s,'Только train, пять академических показателей. Target исключён. ¹ Среднее число дисциплин',645);
}
// 13. Native confusion matrix with consistent class names and all nine counts.
{
 const s=slide(13);
 text(s,'Истинный класс по строкам, прогноз по столбцам',64,160,1152,40,27,C.muted);
 const classes=['Dropout','Enrolled','Graduate'];
 const t=table(s,[['Истина / прогноз',...classes],...data.confusion.map((row,i)=>[classes[i],...row.map(String)])],64,235,800,310,[1.7,1,1,1],{fontSize:30,headerSize:24});
 for(let i=0;i<3;i++) {t.getCell(i+1,i+1).fill='#D5EBED';t.getCell(i+1,0).text.style={typeface:FONT,fontSize:27,color:classColors[i],bold:true};}
 text(s,'Recall',934,237,282,42,29,C.muted);
 text(s,'72,43%',934,304,282,60,43,C.orange,true);
 text(s,'Dropout',934,365,282,40,27,C.orange);
 text(s,'51,26%',934,440,282,60,43,C.blue,true);
 text(s,'Enrolled',934,501,282,40,27,C.blue);
 text(s,'484 верных ответа, 180 ошибок',64,577,800,44,30,C.teal,true);
 foot(s,'Выбранная ANN, test n = 664. Описательные срезы не подтверждают fairness');
}
// 14. Conclusions and proposal are clearly separated from completed experiments.
{
 const s=slide(14);
 text(s,'Академические результаты\nповышают качество LR',64,171,640,102,36,C.teal,true);
 text(s,'Выбор ANN по validation\nсохраняется после test',64,327,640,90,31);
 text(s,'Ограничения',796,171,420,50,31,C.orange,true);
 text(s,'Один вуз и одно разбиение\nНет индивидуальных дат\nНужна внешняя проверка',796,247,420,160,28);
 text(s,'Следующий эксперимент',64,479,1152,46,31,C.teal,true);
 text(s,'Повторные folds для трёх seed, устойчивость ANN / LR\nи чувствительность к демографическим признакам',64,544,1152,86,28);
 foot(s,'Research proposal описывает будущую работу. Применение предполагает участие человека',650);
}
// 15. Short selected bibliography, full references and Q&A in the notes.
{
 const s=slide(15);
 const refs=[
  ['Realinho et al. (2021, 2022)','UCI dataset и статья о данных','10.24432/C5MC89   /   10.3390/data7110146'],
  ['Glandorf et al. (2024)','Временная изменчивость прогноза','10.1145/3636555.3636906'],
  ['Cawley & Talbot (2010)','Model selection и риск переобучения','jmlr.org/papers/v11/cawley10a.html'],
  ['Setiawan et al. (2026)','Обзор многоклассовых образовательных задач','10.7160/eriesj.2026.190106'],
 ];
 for(let i=0;i<refs.length;i++) {
  const y=166+i*96;
  text(s,refs[i][0],64,y,390,36,27,C.teal,true);
  text(s,refs[i][1],475,y,741,35,25);
  text(s,refs[i][2],475,y+41,741,30,21,C.muted);
 }
 text(s,'Полная библиография: 26 источников в отчёте',64,576,1152,40,27);
 text(s,'AI assistance: Codex. Команда проверяет и объясняет материалы',64,631,1152,32,20,C.muted);
 s.speakerNotes.text+='\n\nВОПРОСЫ К ЗАЩИТЕ, НЕ ЗАЧИТЫВАТЬ:\n'+data.qa.map((q,i)=>`${i+1}. ${q.question}\n${q.answer}`).join('\n\n');
}

if(deck.slides.items.length!==15) throw new Error('Expected exactly 15 slides');
const candidatePath=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await fs.writeFile(path.join(BUILD,'presentation.proto.json'),JSON.stringify(deck.toProto()));
for(const [i,s] of deck.slides.items.entries()) {
 const png=await deck.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await s.export({format:'layout'});
 await fs.writeFile(path.join(BUILD,`slide-${String(i+1).padStart(2,'0')}.layout.json`),await layout.text());
}
const finalPath=path.join(OUT,process.env.PRESENTATION_FILENAME??'student_outcomes_presentation.pptx');
const result=await finalizePresentation({
 workspaceDir:ROOT,candidatePath,finalPath,
 pythonExecutable:path.join(RUNTIME,'dependencies/python/bin/python3'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',
  ...[7,12,13].flatMap(number=>['--require-native-table-slide',String(number)])],
 explicitTotalSlideCount:15,
 requiredNativeTableOwnerSlides:[7,12,13],requiredNativeChartOwnerSlides:[3,6,9,10,11,12],
 materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,fontPolicy,
 receiptPath:path.join(BUILD,`${path.basename(finalPath)}.validation.json`),
});
console.log(JSON.stringify({finalPath,slides:15,sha256:result.finalSha256,
  nativeCharts:result.packageIntegrity.chart_count,receipt:result.receiptPath},null,2));
